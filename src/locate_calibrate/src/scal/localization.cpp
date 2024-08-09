// Copyright (c) <2024>, <Hu Nan University> All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "localization.h"
#include <boost/format.hpp>
double cal_sum(vector<double> data)
{
    double res,tmp = 0;
    for (size_t i = 0; i < data.size(); i++)
    {
        tmp +=data[i];
    }
    res = tmp / data.size();
    return res;

}

double cal_var(double x1,double x2)
{
    double res,mean;
    // mean = (x1 + x2)/2;
    // res = (pow((x1 - mean),2) + pow((x2 - mean),2))/2;

    res = fabs(x1 - x2);
    if(res == 0)
        res += 0.0001;
    return res;
}
#define stateSize 1
#define measSize 1
#define controlSize 0
#define hypothesis_anchorNUM 12
#define anchor_num 6
#define PI 3.1415926
KalmanFilter kf[4] = {KalmanFilter(stateSize, measSize, controlSize),
                KalmanFilter(stateSize, measSize, controlSize),
                KalmanFilter(stateSize, measSize, controlSize),
                KalmanFilter(stateSize, measSize, controlSize)};
Eigen::MatrixXd A(stateSize, stateSize);
Eigen::MatrixXd B(1,1);
Eigen::MatrixXd H(measSize, stateSize);
Eigen::MatrixXd P(stateSize, stateSize);
Eigen::MatrixXd R(measSize, measSize);
Eigen::MatrixXd Q(stateSize, stateSize);
int flag[6] = {0,0,0,0} ;
Eigen::VectorXd x(stateSize);
Eigen::VectorXd z(measSize);
Eigen::VectorXd res(stateSize);
double last_distance[6] = {-1,-1,-1,-1,-1,-1};
vector<vector<double>> distance_window(6);
std::mutex odom_mutex;
std::mutex imu_mutex;

int find_match(double target_time,std::vector<nav_msgs::Odometry> data)
{
    float threhold = 0.5;
    int best_index = 999;
    double best = 999;
    for (size_t i = 0; i < data.size(); i++)
    {
        double tmp = abs(data[i].header.stamp.toSec() - target_time);
        if(tmp < best)
        {
            best = tmp;
            best_index = i;
        }
    }
    if(best > threhold)
        return 999;
    else
        return best_index;
}

int find_match(double target_time,std::vector<sensor_msgs::Imu> data)
{
    float threhold = 0.5;
    int best_index = 999;
    double best = 999;
    for (size_t i = 0; i < data.size(); i++)
    {
        double tmp = abs(data[i].header.stamp.toSec() - target_time);
        if(tmp < best)
        {
            best = tmp;
            best_index = i;
        }
    }
    if(best > threhold)
        return 999;
    else
        return best_index;
}

Localization::Localization(ros::NodeHandle n)
{
    pose_realtime_pub = n.advertise<geometry_msgs::PoseStamped>("realtime/pose", 1);

    pose_optimized_pub = n.advertise<geometry_msgs::PoseStamped>("optimized/pose", 1);

    path_optimized_pub = n.advertise<nav_msgs::Path>("optimized/path", 1);

    number_measurements = 0;


    A << 1;
    B << 1;
    H << 1;
    P.setIdentity();
    R<<0;
    Q<<0.001;
    z.setZero();

// For g2o optimizer
    solver = new Solver();

    solver->setBlockOrdering(false);

    se3blockSolver = new SE3BlockSolver(solver);

    optimizationsolver = new g2o::OptimizationAlgorithmLevenberg(se3blockSolver);

    optimizer.setAlgorithm(optimizationsolver);

    g2o::ParameterSE3Offset* zero_offset = new g2o::ParameterSE3Offset;
    zero_offset->setId(0);
    optimizer.addParameter(zero_offset);

    bool verbose_flag;
    if(n.param("optimizer/verbose", verbose_flag, false))
    {
        ROS_WARN("Using optimizer verbose flag: %s", verbose_flag ? "true":"false");
        optimizer.setVerbose(verbose_flag);
    }

    if(n.param("optimizer/maximum_iteration", iteration_max, 20))
        ROS_WARN("Using optimizer maximum iteration: %d", iteration_max);

    if(n.param("optimizer/minimum_optimize_error", minimum_optimize_error, 1000.0))
        ROS_WARN("Will skip estimation if optimization error is larger than: %f", minimum_optimize_error);

// For robots
    if(n.getParam("robot/trajectory_length", trajectory_length))
        ROS_WARN("Using robot trajectory_length: %d", trajectory_length);

    if(n.param("robot/maximum_velocity", robot_max_velocity, 1.0))
        ROS_WARN("Using robot maximum_velocity: %fm/s", robot_max_velocity);

    if(n.param("robot/distance_outlier", distance_outlier, 1.0))
        ROS_WARN("Using uwb outlier rejection distance: %fm", distance_outlier);


// For UWB initial position parameters reading
    if(!n.getParam("/uwb/nodesId", nodesId))
        ROS_ERROR("Can't get parameter nodesId from UWB");

    if(!n.getParam("/uwb/nodesPos", nodesPos))
        ROS_ERROR("Can't get parameter nodesPos from UWB");
    self_id = nodesId.back();
    std::cout<<self_id<<std::endl;
    // nodesId.pop_back();
    std::cout<<self_id<<std::endl;
    ROS_WARN("Init self robot ID: %d with moving option", self_id);

    for (size_t i = 0; i < nodesId.size(); ++i)
    {
        if(n.hasParam("topic/relative_range")||self_id==nodesId[i])
        {
            robots.emplace(nodesId[i], Robot(nodesId[i], false, trajectory_length));
            ROS_WARN("robot ID %d is set moving", nodesId[i]);
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose(0,3) = nodesPos[i*3];
            pose(1,3) = nodesPos[i*3+1];
            pose(2,3) = nodesPos[i*3+2];
            robots.at(nodesId[i]).init(optimizer, pose);
            ROS_WARN("Init robot ID: %d with position (%.2f,%.2f,%.2f)", nodesId[i], pose(0,3), pose(1,3), pose(2,3));
        }
        else // for fixed anchor
        {
            for (size_t j = 0; j < hypothesis_anchorNUM; j++)
            {
                int robot_id = nodesId[i]*hypothesis_anchorNUM + j;
                robots.emplace(robot_id, Robot(robot_id, false, 1));
                // ROS_WARN("robot ID %d is set anchor %d", robot_id,nodesId[i]);
                float ax = nodesPos[i*3],ay = nodesPos[i*3+1],az =nodesPos[i*3+2];
                float tx = nodesPos[anchor_num*3],ty = nodesPos[anchor_num*3+1],tz = nodesPos[anchor_num*3+2];
                float dis = sqrt(pow((ax-tx),2)+pow((ay-ty),2));
                float theta = 2*PI/hypothesis_anchorNUM *j;
                float x = tx + cos(theta) * dis;
                float y = ty + sin(theta) * dis;
                Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
                pose(0,3) = x;
                pose(1,3) = y;
                pose(2,3) = tz;
                if(j ==0)
                {
                    pose(0,3) = ax;
                    pose(1,3) = ay;
                    pose(2,3) = az;
                }
                robots.at(robot_id).init(optimizer, pose);
                ROS_WARN("Init robot anchor ID: %d with position (%.2f,%.2f,%.2f)", robot_id, pose(0,3), pose(1,3), pose(2,3));
            }
        }
    }



// For Debug
    if(n.getParam("log/filename_prefix", name_prefix))
        if(antennaOffset.size() > 0)
            set_file(antennaOffset);
        else
            set_file();
    else
        ROS_WARN("Won't save any log files.");

    if(n.param<string>("frame/target", frame_target, "estimation"))
        ROS_WARN("Using topic target frame: %s", frame_target.c_str());

    if(n.param<string>("frame/source", frame_source, "local_origin"))
        ROS_WARN("Using topic source frame: %s", frame_source.c_str());

    if(n.param<bool>("publish_flag/tf", publish_tf, false))
        ROS_WARN("Using publish_flag/tf: %s", publish_tf ? "true":"false");

    if(n.param<bool>("publish_flag/range", publish_range, false))
        ROS_WARN("Using publish_flag/range: %s", publish_range ? "true":"false");

    if(n.param<bool>("publish_flag/pose", publish_pose, false))
        ROS_WARN("Using publish_flag/pose: %s", publish_pose ? "true":"false");

    if(n.param<bool>("publish_flag/twist", publish_twist, false))
        ROS_WARN("Using publish_flag/twist: %s", publish_twist ? "true":"false");

	if(n.param<bool>("publish_flag/lidar", publish_lidar, false))
        ROS_WARN("Using publish_flag/lidar: %s", publish_lidar ? "true":"false");

    if(n.param<bool>("publish_flag/imu", publish_imu, false))
        ROS_WARN("Using publish_flag/imu: %s", publish_imu ? "true":"false");

    if(n.param<bool>("publish_flag/relative_range", publish_relative_range, false))
        ROS_WARN("Using publish_flag/relative_range: %s", publish_relative_range ? "true":"false");

}


void Localization::solve()
{
    timer.tic();

    optimizer.initializeOptimization();

    optimizer.optimize(iteration_max);

    auto edges = optimizer.activeEdges();
    if(edges.size()>100)
    {
        for(auto edge:edges)
            if (edge->chi2() > 2.0 && edge->dimension () ==1)
            {
                edge->setLevel(1);
                ROS_WARN("Removed one Range Edge");
            }
    }
    ROS_INFO("Graph optimized with error: %f", optimizer.chi2());

    g2o::SparseBlockMatrix<MatrixXd> spinv;

    if(optimizer.computeMarginals(spinv, robots.at(self_id).last_vertex()))
        cout<<spinv.block(0,0)<<endl;
    else
        cout<<"can't compute"<<endl;

    timer.toc();
}


void Localization::publish()
{
    double error = optimizer.chi2();

    if (error < minimum_optimize_error)
        ROS_INFO("Graph optimized with error: %f ", error);
    else
    {
        ROS_WARN("Skip optimization with error: %f ", error);
        return;
    }


    auto pose = robots.at(self_id).current_pose();

    pose.header.frame_id = frame_source;

    pose_realtime_pub.publish(pose);

    auto path = robots.at(self_id).vertices2path();

    path->header.frame_id = frame_source;

    path_optimized_pub.publish(*path);

    pose_optimized_pub.publish(path->poses[trajectory_length/2]);

    // auto pose = robots.at(0).current_pose();

    // std::cout << "flag_save_file:" << flag_save_file <<std::endl;




    if(flag_save_file)
    {
        save_file(pose, realtime_filename);
        save_file(path->poses[trajectory_length/2], optimized_filename);
        std::ofstream out(anchor_filename,std::ios::app);

        out<<fixed<<setprecision(9)<<pose.header.stamp.toSec();

        for (size_t i = 0; i < (nodesId.size()-1)*hypothesis_anchorNUM; i++)
        {
            auto anchor_pos = robots.at(i).current_pose();
            out<<" "<<anchor_pos.pose.position.x<<" "<<anchor_pos.pose.position.y<<" "<<anchor_pos.pose.position.z;
        }
        out<<std::endl;
        out.close();
    }

    if(publish_tf)
    {
        // cout << "###########publish_tf#############"<<endl;
        if(publish_relative_range)
        {
            for (size_t i = 0; i < nodesId.size(); ++i)
            {
                auto pose = robots.at(nodesId[i]).current_pose();

                pose.header.frame_id = frame_source;

                tf::poseMsgToTF(pose.pose, transform);

                br.sendTransform(tf::StampedTransform(transform, pose.header.stamp, frame_source, "rl_" + std::to_string(nodesId[i])));
            }
        }
        else
        {
            tf::poseMsgToTF(pose.pose, transform);
            br.sendTransform(tf::StampedTransform(transform, pose.header.stamp, frame_source, frame_target));
        }

    }

}
geometry_msgs::PoseWithCovarianceStamped toPoseWithCovarianceStamped(const Eigen::Isometry3d& pose, const ros::Time& stamp, const std::string& frame_id) {
    geometry_msgs::PoseWithCovarianceStamped pose_msg;


    pose_msg.pose.pose.position.x = pose.translation().x();
    pose_msg.pose.pose.position.y = pose.translation().y();
    pose_msg.pose.pose.position.z = pose.translation().z();

    Eigen::Quaterniond q(pose.rotation());
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    // Set covariance to zero as it's not provided by Eigen::Isometry3d
    for (int i = 0; i < pose_msg.pose.covariance.size(); i++) {
        pose_msg.pose.covariance[i] = 0.0;
    }

    return pose_msg;
}

void Localization::addPoseEdge(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose_cov_)
{
    geometry_msgs::PoseWithCovarianceStamped pose_cov(*pose_cov_);

    if (pose_cov.header.frame_id != robots.at(self_id).last_header(sensor_type.pose).frame_id)
        key_vertex = robots.at(self_id).last_vertex(sensor_type.pose);

    auto new_vertex = robots.at(self_id).new_vertex(sensor_type.pose, pose_cov.header, optimizer);

    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();

    edge->vertices()[0] = key_vertex;

    edge->vertices()[1] = new_vertex;

    Eigen::Isometry3d measurement;

    tf::poseMsgToEigen(pose_cov.pose.pose, measurement);

    edge->setMeasurement(measurement);

    Eigen::Map<Eigen::MatrixXd> covariance(pose_cov.pose.covariance.data(), 6, 6);

    edge->setInformation(covariance.inverse());

    edge->setRobustKernel(new g2o::RobustKernelCauchy());

    optimizer.addEdge(edge);

    ROS_INFO("added pose edge id: %d frame_id: %s;", pose_cov.header.seq, pose_cov.header.frame_id.c_str());

    // if (publish_pose)
    // {
    //     solve();
    //     publish();
    // }
}


void Localization::addRangeEdge(const nlink_parser::LinktrackNodeframe2rostime &msg)
{
    printf("Begin all\n");
    Eigen::VectorXd solutionMDS;
    int MDS_flag = 0;
    int window_size = 10;
    ++number_measurements;
    int id;
    int index = -1;
    ros::Time time_stamp;
    time_stamp = msg.header.stamp;
    double distance,tmp_R,tmp_rssi = -9999;
    auto myheader = msg.header;
    // myheader.stamp = msg.header.stamp;
    myheader.frame_id = "uwb";
    cout<<fixed<<setprecision(9)<<msg.header.stamp.toSec()<<endl;

    auto vertex_last_requester = robots.at(self_id).last_vertex();

    odom_mutex.lock();
    int last_index = find_match(last_time,odom_list);
    int curt_index = find_match(curt_time,odom_list);
    
/* ---------------------------------------------
*             first stage MDS for anchor position
*/ ---------------------------------------------

    if(number_measurements >=4)
    {
        MDS_flag = 0;
        if(odom_windows.size()<trajectory_length)
        {
            odom_windows.push_back(odom_list[curt_index]);
        }
        else
        {
            odom_windows.erase(odom_windows.begin());
            odom_windows.push_back(odom_list[curt_index]);
        }
        std::vector<double> R;
        for(int i = 0;i<4;i++)
        {
            if(msg.nodes[i].role == 1)
            {
                id = msg.nodes[i].id;
                distance = msg.nodes[i].dis;
                R.push_back(distance);
            }
            else
            {
                R.push_back(-1);
            }
        }
        if (range_windows.size()<trajectory_length)
        {
            range_windows.push_back(R);
        }
        else
        {
            range_windows.erase(range_windows.begin());
            range_windows.push_back(R);
        }
        Ranges_ = convertToEigenMatrix(range_windows);
        RobotPoses_ = convertOdomToEigenMatrix(odom_windows);
        solutionMDS  = MDS_loc_engine_3D(RobotPoses_, Ranges_);
        MDS_flag = 1;
    }




/* ---------------------------------------------
*             second stage optimize
*/ ---------------------------------------------


    //odom factor
    double last_time = robots.at(self_id).last_header().stamp.toSec();
    double curt_time = msg.header.stamp.toSec();
    double dt_requester = curt_time - last_time;
    double cov_requester = pow(robot_max_velocity*dt_requester/3, 2); //3 sigma priciple
    auto vertex_requester = robots.at(self_id).new_vertex(sensor_type.range,myheader, optimizer);
    if(odom_list.size()>1 && last_index!=999 && curt_index!=999)
    {
        double ss = sqrt(pow((odom_list[last_index].pose.pose.position.x - odom_list[curt_index].pose.pose.position.x),2) + pow((odom_list[last_index].pose.pose.position.y - odom_list[curt_index].pose.pose.position.y),2));
        auto edge_requester_range = create_range_edge(vertex_last_requester, vertex_requester, ss, 1);
        odom_list.erase(odom_list.begin(), odom_list.begin() + curt_index+1);
        std::cout << "add Odom edge: "<<ss<<","<<last_index<<","<<curt_index<<std::endl;
        optimizer.addEdge(edge_requester_range);
    }
    else
    {
        auto edge_requester_range = create_range_edge(vertex_last_requester, vertex_requester, 0, cov_requester);
        optimizer.addEdge(edge_requester_range);
    }
    odom_mutex.unlock();

    //set MdS estimate
    if(MDS_flag)
    {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose(0,3) = solutionMDS(0);
        pose(1,3) = solutionMDS(1);
        pose(2,3) = solutionMDS(2);

        //MDS factor
        posewithco = toPoseWithCovarianceStamped(pose);  
        addPoseEdge(posewithco);
    }
    

    //IMU factor
    int curt_imu_index = find_match(curt_time,imu_list);
    if(imu_list.size()>1 && curt_imu_index!=999)
    {
        addImuFactor(imu_list[curt_imu_index]);
    }

    

    float threhold = 0.06;
    double distance_cov = pow(1,2);
    for(int i = 0;i<4;i++)
    {
        if(msg.nodes[i].role == 1)
        {
            id = msg.nodes[i].id;

            double distance_estimation= (robots.at(self_id).last_vertex()->estimate().translation() -
                                        robots.at(msg.nodes[i].id*hypothesis_anchorNUM).last_vertex()->estimate().translation()).norm();
            distance = msg.nodes[i].dis;
            if (number_measurements > trajectory_length && abs(distance_estimation-msg.nodes[i].dis) > distance_outlier)
            {
                ROS_WARN("Reject ID: %d measurement: %fm", msg.nodes[i].id, msg.nodes[i].dis);
                continue;
            }
            else
            {
                distance_window[id].push_back(distance);
                // cout << distance_window[id].size() << endl;
                if(distance_window[id].size() < window_size)
                {


                }
                else if(distance_window[id].size() >= window_size)
                {
                    distance_window[id].erase(distance_window[id].begin());
                    distance = cal_sum(distance_window[id]);
                    // distance_window[id].pop_back();
                    // distance_window[id].push_back(distance);
                    if(last_distance[id] != -1)
                    {
                        if(distance - last_distance[id]>threhold )
                        {
                            distance = last_distance[id] + threhold;
                            distance_window[id].pop_back();
                            distance_window[id].push_back(distance);
                        }
                        else if(distance - last_distance[id] <-threhold)
                        {
                            distance = last_distance[id] - threhold;
                            distance_window[id].pop_back();
                            distance_window[id].push_back(distance);
                        }
                    }
                    last_distance[id] = distance;
                    // distance = accumulate(distance_window[id].begin(),distance_window[id].end(),0) / window_size;
                }

            }
            // out<<fixed<<setprecision(9)<<time_stamp<<" "<<id<<" "<<distance<<std::endl;

            for (size_t j = 0; j < hypothesis_anchorNUM; j++)
            {
                index = id*hypothesis_anchorNUM + j;
                auto vertex_last_responder = robots.at(index).last_vertex();
                auto vertex_responder = robots.at(index).new_vertex(sensor_type.range, myheader, optimizer);
                auto frame_id = robots.at(self_id).last_header().frame_id;
                if((frame_id.find(myheader.frame_id)!=string::npos) || (frame_id.find("none")!=string::npos))
                {
                    auto edge = create_range_edge(vertex_requester, vertex_responder,  distance, distance_cov);
                    optimizer.addEdge(edge);
                    // cout << "added two requester range edge on id: "<<index<<","<<distance<<","<<distance_cov<<endl;
                    // ROS_INFO("added two requester range edge on id: <%d> ", msg.nodes[i].id);
                }
                else
                {
                    auto edge = create_range_edge(vertex_last_requester, vertex_responder, distance, distance_cov + cov_requester);

                    optimizer.addEdge(edge); // decrease computation

                    ROS_INFO("added requester edge with id: <%d>", index);
                }
            }
        }
    }
    if (publish_range && number_measurements > 5)
    {
        solve();
        publish();
    }
}
//derived radial velocity factor
void Localization::addTwistEdge(const geometry_msgs::TwistWithCovarianceStamped::ConstPtr& twist_)
{
    geometry_msgs::TwistWithCovarianceStamped twist(*twist_);

    double dt = twist.header.stamp.toSec() - robots.at(self_id).last_header().stamp.toSec();

    auto last_vertex = robots.at(self_id).last_vertex();

    auto new_vertex = robots.at(self_id).new_vertex(sensor_type.twist, twist.header, optimizer);

    auto edge = create_se3_edge_from_twist(last_vertex, new_vertex, twist.twist, dt);

    optimizer.addEdge(edge);

    ROS_INFO("added twist edge id: %d", twist.header.seq);

    if (publish_twist)
    {
        solve();
        publish();
    }
}

void Localization::addOdomEdge(const nav_msgs::Odometry::ConstPtr& odom_)
{
    nav_msgs::Odometry tmp;
    tmp.header = odom_->header;
    tmp.pose = odom_->pose;

    odom_mutex.lock();
    odom_list.push_back(tmp);
    std::cout << "Get Odom at: "<<tmp.header.stamp.toSec()<<" current list size: "<<odom_list.size()<<std::endl;
    odom_mutex.unlock();

}

void Localization::addImuEdge(const sensor_msgs::Imu::ConstPtr& Imu_)
{
    sensor_msgs::Imu tmp;
    tmp.header = Imu_->header;
    tmp.orientation = Imu_->orientation;
    tmp.orientation_covariance = Imu_->orientation_covariance;
    tmp.angular_velocity = Imu_->angular_velocity;
    tmp.angular_velocity_covariance = Imu_->angular_velocity_covariance;
    tmp.linear_acceleration = Imu_->linear_acceleration;
    tmp.linear_acceleration_covariance = Imu_->linear_acceleration_covariance;

    imu_mutex.lock();
    imu_list.push_back(tmp);
    std::cout << "Get Imu at: "<<tmp.header.stamp.toSec()<<" current list size: "<<imu_list.size()<<std::endl;
    imu_mutex.unlock();
}



void Localization::configCallback(localization::localizationConfig &config, uint32_t level)
{
    ROS_WARN("Get publish_optimized_poses: %s", config.publish_optimized_poses? "ture":"false");

    if (config.publish_optimized_poses)
    {
        ROS_WARN("Publishing Optimized poses");

        auto path = robots.at(self_id).vertices2path();

        for (int i = trajectory_length/2; i < trajectory_length; ++i)
        {
            pose_optimized_pub.publish(path->poses[i]);

            usleep(10000);
        }
        ROS_WARN("Published. Done");
    }
}

void Localization::addImuFactor(const sensor_msgs::Imu::ConstPtr& Imu_)
{
    if (robots.at(self_id).last_header().frame_id.find(Imu_->header.frame_id) == string::npos)
    {
        robots.at(self_id).append_last_header(Imu_->header.frame_id);

        auto last_vertex = robots.at(self_id).last_vertex(sensor_type.range);

        Eigen::Isometry3d current_pose = Eigen::Isometry3d::Identity();

        current_pose.rotate(Quaterniond(Imu_->orientation.w, Imu_->orientation.x, Imu_->orientation.y, Imu_->orientation.z));

        current_pose.translation() = last_vertex->estimate().translation();

        last_vertex->setEstimate(current_pose);

        Eigen::MatrixXd  information = Eigen::MatrixXd::Zero(6,6);
        information(3,3)= 1.0/Imu_->orientation_covariance[0];
        information(4,4)= 1.0/Imu_->orientation_covariance[4];
        information(5,5)= 1.0/Imu_->orientation_covariance[8];// roll, pitch, yaw
        
        g2o::EdgeSE3Prior* edgeprior = new g2o::EdgeSE3Prior();
        edgeprior->setInformation(information);
        edgeprior->vertices()[0]= last_vertex; 
        edgeprior->setMeasurement(current_pose); 
        edgeprior->setParameterId(0,0);
        optimizer.addEdge(edgeprior);

        ROS_INFO("added IMU edge id: %d", Imu_->header.seq);
    }


}

inline Eigen::Isometry3d Localization::twist2transform(geometry_msgs::TwistWithCovariance& twist, Eigen::MatrixXd& covariance, double dt)
{
    tf::Vector3 translation, euler;

    tf::vector3MsgToTF(twist.twist.linear, translation);

    tf::vector3MsgToTF(twist.twist.angular, euler);

    tf::Quaternion quaternion;

    quaternion.setRPY(euler[0]*dt, euler[1]*dt, euler[2]*dt);

    tf::Transform transform(quaternion, translation * dt);

    Eigen::Isometry3d measurement;

    tf::transformTFToEigen(transform, measurement);

    Eigen::Map<Eigen::MatrixXd> cov(twist.covariance.data(), 6, 6);

    covariance = cov*dt*dt;

    return measurement;
}


inline g2o::EdgeSE3* Localization::create_se3_edge_from_twist(g2o::VertexSE3* vetex1, g2o::VertexSE3* vetex2, geometry_msgs::TwistWithCovariance& twist, double dt)
{
    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();

    edge->vertices()[0] = vetex1;

    edge->vertices()[1] = vetex2;

    Eigen::MatrixXd covariance;

    auto measurement = twist2transform(twist, covariance, dt);

    edge->setMeasurement(measurement);

    edge->setInformation(covariance.inverse());

    edge->setRobustKernel(new g2o::RobustKernelCauchy());

    return edge;
}


inline g2o::EdgeSE3Range* Localization::create_range_edge(g2o::VertexSE3* vertex1, g2o::VertexSE3* vertex2, double distance, double covariance)
{
    auto edge = new g2o::EdgeSE3Range();

    edge->vertices()[0] = vertex1;

    edge->vertices()[1] = vertex2;

    edge->setMeasurement(distance);

    Eigen::MatrixXd covariance_matrix = Eigen::MatrixXd::Zero(1, 1);

    covariance_matrix(0,0) = covariance;

    edge->setInformation(covariance_matrix.inverse());

    edge->setRobustKernel(new g2o::RobustKernelCauchy());

    return edge;
}

inline void Localization::save_file(geometry_msgs::PoseStamped pose, string filename)
{
    file.open(filename.c_str(), ios::app);
    file<<boost::format("%.9f") % (pose.header.stamp.toSec())<<" "
        <<pose.pose.position.x<<" "
        <<pose.pose.position.y<<" "
        <<pose.pose.position.z<<" "
        <<pose.pose.orientation.x<<" "
        <<pose.pose.orientation.y<<" "
        <<pose.pose.orientation.z<<" "
        <<pose.pose.orientation.w<<endl;
    file.close();
}


void Localization::set_file()
{
    flag_save_file = true;
    char s[30];
    struct tm tim;
    time_t now;
    now = time(NULL);
    tim = *(localtime(&now));
    strftime(s,30,"_%Y_%b_%d_%H_%M_%S.txt",&tim);
    realtime_filename = name_prefix+"_realtime" + string(s);
    optimized_filename = name_prefix+"_optimized" + string(s);
    anchor_filename = name_prefix+"_anchor" + string(s);

    file.open(realtime_filename.c_str(), ios::trunc|ios::out);
    file<<"# "<<"iteration_max:"<<iteration_max<<"\n";
    file<<"# "<<"trajectory_length:"<<trajectory_length<<"\n";
    file<<"# "<<"maximum_velocity:"<<robot_max_velocity<<"\n";
    file.close();

    file.open(optimized_filename.c_str(), ios::trunc|ios::out);
    file<<"# "<<"iteration_max:"<<iteration_max<<"\n";
    file<<"# "<<"trajectory_length:"<<trajectory_length<<"\n";
    file<<"# "<<"maximum_velocity:"<<robot_max_velocity<<"\n";
    file.close();

    ROS_WARN("Loging to file: %s",realtime_filename.c_str());
    ROS_WARN("Loging to file: %s",optimized_filename.c_str());
}

void Localization::set_file(std::vector<double> antennaOffset)
{
    flag_save_file = true;
    char s[30];
    struct tm tim;
    time_t now;
    now = time(NULL);
    tim = *(localtime(&now));
    strftime(s,30,"_%Y_%b_%d_%H_%M_%S.txt",&tim);
    realtime_filename = name_prefix+"_realtime" + string(s);
    optimized_filename = name_prefix+"_optimized" + string(s);

    file.open(realtime_filename.c_str(), ios::trunc|ios::out);
    file<<"# "<<"iteration_max:"<<iteration_max<<"\n";
    file<<"# "<<"trajectory_length:"<<trajectory_length<<"\n";
    file<<"# "<<"maximum_velocity:"<<robot_max_velocity<<"\n";
    file.close();

    file.open(optimized_filename.c_str(), ios::trunc|ios::out);
    file<<"# "<<"iteration_max:"<<iteration_max<<"\n";
    file<<"# "<<"trajectory_length:"<<trajectory_length<<"\n";
    file<<"# "<<"maximum_velocity:"<<robot_max_velocity<<"\n";
    file.close();

    file.open(optimized_filename.c_str(), ios::trunc|ios::out);
    file<<"# "<<"antenna offsets: ";
    for(unsigned int i = 0; i < antennaOffset.size() - 1; i++)
        file << antennaOffset[i] << ",";
    file << antennaOffset[antennaOffset.size()-1] << "\n";
    file.close();

    ROS_WARN("Loging to file: %s",realtime_filename.c_str());
    ROS_WARN("Loging to file: %s",optimized_filename.c_str());
}

Localization::~Localization()
{
    if (flag_save_file)
    {
        auto path = robots.at(self_id).vertices2path();
        for (int i = trajectory_length/2; i < trajectory_length; ++i)
            save_file(path->poses[i], optimized_filename);
        cout<<"Results Loged to file: "<<optimized_filename<<endl;
    }
}
