# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test_nsg_search.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test_nsg_search.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test_nsg_search.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test_nsg_search.dir/flags.make

tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o: tests/CMakeFiles/test_nsg_search.dir/flags.make
tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o: ../tests/test_nsg_search.cpp
tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o: tests/CMakeFiles/test_nsg_search.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o"
	cd /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o -MF CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o.d -o CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o -c /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/tests/test_nsg_search.cpp

tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.i"
	cd /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/tests/test_nsg_search.cpp > CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.i

tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.s"
	cd /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/tests/test_nsg_search.cpp -o CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.s

# Object files for target test_nsg_search
test_nsg_search_OBJECTS = \
"CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o"

# External object files for target test_nsg_search
test_nsg_search_EXTERNAL_OBJECTS =

tests/test_nsg_search: tests/CMakeFiles/test_nsg_search.dir/test_nsg_search.cpp.o
tests/test_nsg_search: tests/CMakeFiles/test_nsg_search.dir/build.make
tests/test_nsg_search: src/libefanna2e.a
tests/test_nsg_search: tests/CMakeFiles/test_nsg_search.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_nsg_search"
	cd /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_nsg_search.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test_nsg_search.dir/build: tests/test_nsg_search
.PHONY : tests/CMakeFiles/test_nsg_search.dir/build

tests/CMakeFiles/test_nsg_search.dir/clean:
	cd /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test_nsg_search.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test_nsg_search.dir/clean

tests/CMakeFiles/test_nsg_search.dir/depend:
	cd /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/tests /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests /home/arc-x10/Projects/HYNIX_NDP/GraphANNS/nsg/build/tests/CMakeFiles/test_nsg_search.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test_nsg_search.dir/depend

