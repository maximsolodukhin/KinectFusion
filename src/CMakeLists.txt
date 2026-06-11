# headers we export are treated as system headers for downstream consumers
# and warnings are suppressed whenever this project is built as a subproject.
if(PROJECT_IS_TOP_LEVEL)
  set(WARNING_GUARD "")
else()
  set(WARNING_GUARD SYSTEM)
endif()

add_subdirectory(lib)
add_subdirectory(app)
