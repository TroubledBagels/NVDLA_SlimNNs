#!/bin/bash

# Make sure the script exits if any command fails
set -e

# Optional: print all commands before executing
set -x

# Source environment variables if needed
# source /path/to/env.sh

# Navigate to the application directory
cd /home/your_app_directory

# Run the main command or open an interactive shell
# Example: Run a script or build tool
# ./build.sh

# Example: Open a bash shell (for dev/debugging)
exec /bin/bash
