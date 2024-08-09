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
#include <iostream>
#include <Eigen/Dense>

Eigen::MatrixXd convertToEigenMatrix(const std::vector<std::vector<double>>& range_windows) {
    int rows = range_windows.size();
    int cols = range_windows[0].size();

    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = range_windows[i][j];
        }
    }

    return matrix.transpose();
}

Eigen::MatrixXd convertOdomToEigenMatrix(const std::vector<nav_msgs::Odometry>& odom_windows) {
    int rows = odom_windows.size();
    int cols = 3;  // x, y, z

    Eigen::MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        matrix(i, 0) = odom_windows[i].pose.pose.position.x;
        matrix(i, 1) = odom_windows[i].pose.pose.position.y;
        matrix(i, 2) = odom_windows[i].pose.pose.position.z;
    }

    return matrix;
}

Eigen::MatrixXd calculate_distance_matrix(Eigen::MatrixXd A, Eigen::MatrixXd B, bool squared=true) 
{
    if (B.rows() == 0) {
        B = A;
    }
    if (B.cols() < 2) {
        B.resize(1, B.size());
    }
    int m = A.rows();
    int n = B.rows();
    assert(A.cols() == B.cols() && "The number of components for vectors in A does not match that of B!");

    Eigen::MatrixXd Adot = (A.array() * A.array()).rowwise().sum().replicate(1, n);
    Eigen::MatrixXd Bdot = (B.array() * B.array()).rowwise().sum().replicate(m, 1);
    Eigen::MatrixXd ABdot = A * B.transpose();
    Eigen::MatrixXd D_squared = Adot + Bdot - 2 * ABdot;

    if (squared) {
        return D_squared;
    } else {
        Eigen::MatrixXd D = D_squared.array().sqrt();
        return D;
    }
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d> Kabsch(Eigen::MatrixXd P, Eigen::MatrixXd Q) {
    assert(P.rows() == Q.rows() && "Shape of P and Q do not match");
    Eigen::Vector3d p0 = P.colwise().mean();
    Eigen::Vector3d q0 = Q.colwise().mean();
    Eigen::MatrixXd P_centered = P.rowwise() - p0.transpose();
    Eigen::MatrixXd Q_centered = Q.rowwise() - q0.transpose();
    Eigen::MatrixXd C = P_centered.transpose() * Q_centered;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    Eigen::Vector3d t = q0 - p0 * R;
    return std::make_tuple(R, t);
}

Eigen::Vector2d Bancroft_loc_engine_3D(Eigen::MatrixXd anchorCoordinates, Eigen::MatrixXd Ranges) {
    Eigen::MatrixXd B(anchorCoordinates.rows(), anchorCoordinates.cols() + Ranges.cols());
    B << anchorCoordinates, -Ranges;
    Eigen::MatrixXd B_ = B.completeOrthogonalDecomposition().pseudoInverse();
    double a = 0.5 * dotLorentz(B, B);
    double c2 = dotLorentz(B_.rowwise().sum(), B_.rowwise().sum());
    double c1 = 2 * (dotLorentz(B_ * a, B_.rowwise().sum()) - 1);
    double c0 = dotLorentz(B_ * a, B_ * a);
    double discriminant = c1 * c1 - 4 * c0 * c2;
    double root1, root2;
    if (discriminant >= 0) {
        root1 = (-c1 + std::sqrt(discriminant)) / (2 * c2);
        root2 = (-c1 - std::sqrt(discriminant)) / (2 * c2);
    } else {
        root1 = -c1 / (2 * c2);
        root2 = root1;
    }

    Eigen::VectorXd u1 = B_ * (a + root1);
    Eigen::VectorXd u2 = B_ * (a + root2);
    Eigen::MatrixXd U(2, u1.size());
    U << u1, u2;
    U = U.rowwise().norm();
    std::sort(U.data(), U.data() + U.size());
    return U.row(0).head(2);
}

Eigen::VectorXd MDS_loc_engine_3D(Eigen::MatrixXd anchorCoordinates, Eigen::MatrixXd Ranges) {
    int n = anchorCoordinates.rows();
    int p = anchorCoordinates.cols();
    int k = Ranges.rows();
    int m = Ranges.cols();
    assert(n == k && "The number of anchors does not match the number of ranges");

    Eigen::MatrixXd D_squared = calculate_distance_matrix(anchorCoordinates, Ranges);
    Eigen::MatrixXd D_full(n + m, n + m);
    D_full << D_squared, Ranges.array().pow(2), (Ranges.transpose().array().pow(2), 0);
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(n + m, n + m) - Eigen::MatrixXd::Ones(n + m, n + m) / (n + m);
    Eigen::MatrixXd B = -0.5 * C * D_full * C;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd P = svd.matrixU().leftCols(p) * svd.singularValues().head(p).cwiseSqrt().asDiagonal();
    Eigen::Matrix3d R, t;
    std::tie(R, t) = Kabsch(P.topRows(P.rows() - 1), anchorCoordinates);
    Eigen::VectorXd out = P.bottomRows(1) * R + t;
    return out;
}

double dotLorentz(Eigen::VectorXd u, Eigen::VectorXd v, int dim = 1) {
    if (u.size() > 1) {
        if (dim == 0) {
            return (u.head(u.size() - 1).transpose() * v.head(v.size() - 1) - u(u.size() - 1) * v(v.size() - 1));
        } else if (dim == 1) {
            return (u.head(u.size() - 1).transpose() * v.head(v.size() - 1) - u(u.size() - 1) * v(v.size() - 1));
        } else {
            return 0;
        }
    } else {
        return (u(u.size() - 1) * v(v.size() - 1) - u.head(u.size() - 1).transpose() * v.head(v.size() - 1));
    }
}

bool checkInitialization(Eigen::MatrixXd anchorCoordinates, Eigen::MatrixXd Ranges, Eigen::VectorXd poseEstimate, double PdopThs = 20, double sigmaThs = 0.1, double planeRatioThs = 1) {
    int n = anchorCoordinates.rows();
    int p = anchorCoordinates.cols();
    PdopThs /= std::sqrt(n);

    Eigen::MatrixXd A = (anchorCoordinates - poseEstimate).array() / Ranges.array();
    Eigen::MatrixXd Q = (A.transpose() * A).inverse();
    double PDOP = std::sqrt(Q.trace());

    Eigen::VectorXd RangesEstimate = (anchorCoordinates - poseEstimate).rowwise().norm();
    double sigma = (RangesEstimate - Ranges).norm() / std::sqrt(n - 1);

    Eigen::VectorXd meanAnchorCoordinates = anchorCoordinates.colwise().mean();
    Eigen::MatrixXd U, S, Vt;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd((anchorCoordinates.rowwise() - meanAnchorCoordinates.transpose()), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd S = svd.singularValues().asDiagonal();
    Eigen::MatrixXd Vt = svd.matrixV().transpose();
    Eigen::MatrixXd V = Vt.transpose();
    Eigen::VectorXd q = poseEstimate - meanAnchorCoordinates;
    double angle = std::asin(std::abs(V.col(V.cols() - 1).transpose() * q) / q.norm());
    double b = -q.norm() * std::sin(angle) + std::sqrt(std::sin(angle) * std::sin(angle) * q.norm() * q.norm() + sigma * (2 * q.norm() + sigma));
    double planeRatio = S(S.size() - 1) / std::sqrt(n) / b;
    return (planeRatio > planeRatioThs && PDOP < PdopThs && sigma < sigmaThs);
}
    
    
    
    
    
    