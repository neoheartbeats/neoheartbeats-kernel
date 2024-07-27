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
    read -rp "Do you agree to the terms and conditions of the license agreement? [Yes/No]:" response
    if [[ ! "${response,,}" =~  ^(yes|y)$ ]]; then
        echo "You did not agree to the license agreement. Exiting."
        exit 1
    fi
}

# Retry download function
download_with_retry() {
    local file_name="$1"
    local download_url="$2"
    local target_file="$3"
    local retry_count=3

    for i in $(seq 1 $retry_count); do
        echo "Attempting to download $file_name (attempt $i / $retry_count)"
        wget -c -q --show-progress "$download_url" -O "$target_file"
        if [[ $? -eq 0 ]]; then
            echo "Download successful"
            return 0
        else
            echo "Download failed, retrying..."
            sleep 5
        fi
    done

    echo "Download failed, exiting"
    return 1
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
    storcli_url="https://downloadmirror.intel.com/820588/StorCLI_007.2705.0000.0000.zip"

    if ! storcli version 2>/dev/null| grep -q "7.2705.00"; then
        echo "Installing storcli 1.20.15..."
        echo "Intel Software License Agreement: "
        echo "http"
        download_with_retry storcli "$storcli_url" "/tmp/storcli.zip"
        unzip -qp /tmp/storcli.zip "*/*.zip" > /tmp/Unified_storcli_all_os.zip
        unzip -q /tmp/Unified_storcli_all_os.zip -d /tmp/storcli

        cd /tmp/storcli/Unified_storcli_all_os/ || exit 1

        if [[ "$package" == "deb" ]]; then
            sudo dpkg -i Ubuntu/storcli_*_all.deb
        elif [[ "$package" == "rpm" ]]; then
            sudo rpm -i Linux/storcli-*.noarch.rpm
        fi
    fi

    # Install sas3ircu 17.00.00.00
    sas3ircu_url="https://example.com/sas3ircu-17.00.00.00.tar.gz"

    if ! sas3ircu | grep -q "17.00.00.00"; then
        echo "Installing sas3ircu 17.00.00.00..."
        download_with_retry sas3ircu "$sas3ircu_url" "/tmp/sas3ircu.tar.gz"
        tar -xzf /tmp/sas3ircu.tar.gz -C /tmp
        sudo cp /tmp/sas3ircu /usr/local/bin/
    fi

    # Check and install other specific versions if necessary
    echo "Post installation checks and specific version installs completed."
}

# Function to display and require agreement
agreement

# Determine the distribution and call the appropriate function
case $(uname -s) in
    Linux|GNU*)
        if [[ -f /etc/os-release || \
            -f /usr/lib/os-release || \
            -f /etc/openwrt_release || \
            -f /etc/lsb-release ]]; then

            # Source the os-release file
            for file in /etc/lsb-release /usr/lib/os-release \
                        /etc/os-release  /etc/openwrt_release; do
                source "$file" && break
            done

            if [[ $ID =~ ^(debian|ubuntu) ]]; then
                echo "Detected Debian/Ubuntu based system."
                package="deb"
                install_debian
            else
                echo "This Linux distribution is not currently supported."
            fi

        elif [ -f /etc/redhat-release ]; then
            echo "Detected Fedora/RHEL/CentOS based system."
            package="rpm"
            install_redhat
        else
            echo "Unsupported distribution. Please install the packages manually."
            exit 1
        fi
    ;;
    *)
        echo "Non-linux systems are not supported."
        exit 1
    ;;
esac

# Perform post-install checks
post_install_checks


if storcli version 2>/dev/null| grep -q "7.2705.00"; then
    echo "All packages installed successfully."
else
    echo "Installation failure."
fi
