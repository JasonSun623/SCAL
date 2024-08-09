ADD_LIBRARY(localization2 STATIC
	${G2O_LIB_TYPE}
  	localization.h
  	localization.cpp
  	robot.cpp
  	robot.h
)

SET_TARGET_PROPERTIES(localization2 PROPERTIES OUTPUT_NAME localization2)

ADD_DEPENDENCIES(localization2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})  

TARGET_LINK_LIBRARIES(localization2
	${catkin_LIBRARIES}
	${CHOLMOD_LIBRARIES}
	${CSPARSE_LIBRARIES}
	types_edge_se3range2
	g2o_core
	g2o_types_slam3d
	g2o_solver_csparse
	g2o_stuff
	g2o_csparse_extension
	# csparse
)

# INSTALL(TARGETS localization
#   RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#   LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# )

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
include_directories( "/usr/include/suitesparse")

# INSTALL(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/g2o/types/sba)
