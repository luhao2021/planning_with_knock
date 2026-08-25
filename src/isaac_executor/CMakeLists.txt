cmake_minimum_required(VERSION 3.8)
project(isaac_executor)

install(PROGRAMS
  scripts/isaac_executor.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES scripts/problem.py DESTINATION lib/${PROJECT_NAME})
