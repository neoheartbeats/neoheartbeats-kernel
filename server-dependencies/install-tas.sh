#!/bin/bash

# Credits:
agreement() {
    echo "--------------------------------------------"
    echo "StorCLI Standalone Utility License Agreement"
    echo "--------------------------------------------"
    echo "This software is provided by Intel Corporation."
    echo "For full details, please visit the following URL:"
    echo "https://www.intel.com/content/www/us/en/download/17809/storcli-standalone-utility.html"
    echo ""
    echo "By using this script, you agree to the terms and conditions of the license agreement."
    echo "Do you agree to the terms and conditions of the license agreement? (yes/no)"
    read -r response
    if [[ "$response" !=  "yes"]]; then
        echo "You did not agree to the license agreement. Exiting."
        exit 1
    fi
}

# Function to install packages on Debian/Ubuntu
install_debian() {
    sudo apt-get update -y
    sudo apt-get install -y ethtool \
         openipmi \
         smartmontools \
         glibc-source \
         storcli \
         mdadm \
         network-manager \
         net-tools \
         lsscsi \
         util-linux \
         sas3ircu \
         bc \
         pciutils \
         fio
}

# Function to install packages on Fedora/RHEL/CentOS
install_redhat() {
    sudo yum check-update
    sudo yum install -y epel-release
    sudo yum install -y ethtool OpenIPMI smartmontools glibc storcli mdadm NetworkManager \
         net-tools lsscsi util-linux sas3ircu bc pciutils fio
}

# Function to check and install specific versions or additional steps if required
post_install_checks() {
    # Install storcli 1.20.15
    if ! storcli version | grep -q "7.2705.00"; then
        echo "Installing storcli 1.20.15..."
        echo "Intel Software License Agreement: "
        echo "http"
        wget https://downloadmirror.intel.com/820588/StorCLI_007.2705.0000.0000.zip -O /tmp/storcli.zip
        unzip /tmp/storcli.zip -d /tmp/storcli
        cd /tmp/storcli/Unified_storcli_all_os/

        if [ -f /etc/debian_version ]; then
            cd Ubuntu/
            sudo dpkg -i storcli_007.2705.0000.0000_all.deb
        elif [ -f /etc/redhat-release ]; then
            cd Linux/
            sudo rpm -i storcli-007.2705.0000.0000-1.noarch.rpm
        fi
    fi

    # Install sas3ircu 17.00.00.00
    if ! sas3ircu | grep -q "17.00.00.00"; then
        echo "Installing sas3ircu 17.00.00.00..."
        wget https://example.com/sas3ircu-17.00.00.00.tar.gz -O /tmp/sas3ircu.tar.gz
        tar -xzf /tmp/sas3ircu.tar.gz -C /tmp
        sudo cp /tmp/sas3ircu /usr/local/bin/
    fi

    # Check and install other specific versions if necessary
    echo "Post installation checks and specific version installs completed."
}

# Function to display and require agreement
agreement

# Determine the distribution and call the appropriate function
if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu based system."
    install_debian
elif [ -f /etc/redhat-release ]; then
    echo "Detected Fedora/RHEL/CentOS based system."
    install_redhat
else
    echo "Unsupported distribution. Please install the packages manually."
    exit 1
fi

# Perform post-install checks
post_install_checks

echo "All packages installed successfully."
