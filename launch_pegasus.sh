#!/bin/bash
#
# launch_pegasus.sh
# Launches the Pegasus Simulator (a drone-simulation extension for NVIDIA Isaac Sim)
# by running one of its standalone example apps through Isaac Sim's bundled Python.
#
# Usage:
#   ./launch_pegasus.sh                # runs the default example (1_px4_single_vehicle.py)
#   ./launch_pegasus.sh 3_ros2_single_vehicle.py
#   ./launch_pegasus.sh /abs/path/to/your_app.py [extra args...]
#
set -e

# --- Paths -------------------------------------------------------------------
ISAACSIM_PATH="${ISAACSIM_PATH:-${HOME}/isaacsim}"
ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-${ISAACSIM_PATH}/python.sh}"
PEGASUS_PATH="${PEGASUS_PATH:-${ISAACSIM_PATH}/PegasusSimulator}"
EXAMPLES_DIR="${PEGASUS_PATH}/examples"

# --- Resolve which script to run --------------------------------------------
DEFAULT_EXAMPLE="1_px4_single_vehicle.py"
TARGET="${1:-${DEFAULT_EXAMPLE}}"
shift || true   # remaining args are forwarded to the python app

# If the target isn't an existing path, look for it inside the examples dir.
if [[ ! -f "${TARGET}" ]]; then
    if [[ -f "${EXAMPLES_DIR}/${TARGET}" ]]; then
        TARGET="${EXAMPLES_DIR}/${TARGET}"
    else
        echo "ERROR: could not find script '${TARGET}'." >&2
        echo "Available examples in ${EXAMPLES_DIR}:" >&2
        ls -1 "${EXAMPLES_DIR}"/*.py 2>/dev/null | xargs -n1 basename >&2
        exit 1
    fi
fi

# --- Sanity checks -----------------------------------------------------------
if [[ ! -x "${ISAACSIM_PYTHON}" ]]; then
    echo "ERROR: Isaac Sim python launcher not found/executable at ${ISAACSIM_PYTHON}" >&2
    exit 1
fi

# --- ROS 2 -------------------------------------------------------------------
# Isaac Sim runs its own Python 3.11, but system ROS (/opt/ros/humble) is built
# for Python 3.10, so its rclpy (_rclpy_pybind11.cpython-310) can't load into
# Kit -> "No module named 'rclpy._rclpy_pybind11'". Isaac ships its OWN humble
# rclpy (cpython-311) inside the ros2 bridge; we just have to point
# LD_LIBRARY_PATH at that bridge's humble/lib so its dependent .so's resolve.
#
# We deliberately do NOT source /opt/ros/humble here: sourcing it sets
# ROS_DISTRO, which makes Isaac's setup_ros_env.sh skip adding the internal lib
# path, AND it leaks Python 3.10 site-packages onto PYTHONPATH. Both sides still
# use rmw_fastrtps_cpp, so `ros2 topic list` from a separately-sourced terminal
# still sees the published topics over DDS. Disable this block with
# PEGASUS_SOURCE_ROS2=0.
ROS2_BRIDGE_LIB="${ISAACSIM_PATH}/exts/isaacsim.ros2.bridge/humble/lib"
if [[ "${PEGASUS_SOURCE_ROS2:-1}" != "0" && -d "${ROS2_BRIDGE_LIB}" ]]; then
    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${ROS2_BRIDGE_LIB}"
    echo "   ROS 2     : internal ${ROS_DISTRO} bridge libs (RMW=${RMW_IMPLEMENTATION})"
fi

# --- Kit / RTX settings ------------------------------------------------------
# This machine runs Canonical driver 535.309.01. NVIDIA packs the Vulkan
# driverVersion minor field into 8 bits, so 309 overflows (309 & 0xFF = 53) and
# Omniverse misreads the driver as "535.53", which it rejects as below the
# minimum (535.129) -> the RTX renderer fails and the viewport stays empty.
# The driver is functionally fine (CUDA/PhysX work), so we skip the version
# check. Set PEGASUS_SKIP_DRIVER_CHECK=0 to disable this workaround.
KIT_ARGS=()
if [[ "${PEGASUS_SKIP_DRIVER_CHECK:-1}" != "0" ]]; then
    KIT_ARGS+=("--/rtx/verifyDriverVersion/enabled=false")
fi

echo "=================================================================="
echo " Launching Pegasus Simulator"
echo "   Isaac Sim : ${ISAACSIM_PATH}"
echo "   Python    : ${ISAACSIM_PYTHON}"
echo "   App       : ${TARGET}"
echo "   Kit args  : ${KIT_ARGS[*]:-<none>}"
echo "=================================================================="

# --- Launch ------------------------------------------------------------------
exec "${ISAACSIM_PYTHON}" "${TARGET}" "${KIT_ARGS[@]}" "$@"
