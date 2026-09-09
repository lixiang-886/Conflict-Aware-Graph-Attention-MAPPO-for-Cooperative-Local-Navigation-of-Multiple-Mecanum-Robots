#!/usr/bin/env bash

setup_file="${1:-}"
if [ -z "${setup_file}" ]; then
  echo "Usage: source scripts/source_setup_file.sh <setup.bash>" >&2
  return 2 2>/dev/null || exit 2
fi

if [ ! -f "${setup_file}" ]; then
  echo "setup file not found: ${setup_file}" >&2
  return 1 2>/dev/null || exit 1
fi

_mrpp_restore_nounset=0
case "$-" in
  *u*)
    _mrpp_restore_nounset=1
    set +u
    ;;
esac

# ROS and colcon setup files may read optional variables that are unset.
# Temporarily disabling nounset keeps our scripts strict without breaking setup.
# shellcheck disable=SC1090
source "${setup_file}"

# Keep ROS/Gazebo experiments on a project-local domain unless the caller
# explicitly selected one. The default DDS domain can be polluted by stale
# system nodes and break ros_gz_bridge discovery.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

if [ "${_mrpp_restore_nounset}" -eq 1 ]; then
  set -u
fi
unset _mrpp_restore_nounset
