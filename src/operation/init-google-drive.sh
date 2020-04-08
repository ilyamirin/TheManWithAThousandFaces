
if ! [ -x "$(command -v rclone)" ]; then
    echo "Installing RClone..."

    curl https://rclone.org/install.sh | sudo bash

    echo "├── Complete"
fi

if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "Configuring connection to Google Drive..."
    echo "Please follow the link and log in as keteride@gmail.com"

    rclone config create KeterideDrive drive config_is_local false
    cp ~/.config/rclone/rclone.conf .rclone.conf

    echo "├── Complete"
fi

echo "Synchronizing resources with Google Drive..."
rclone sync KeterideDrive:Financial-Analytics-Classifier src/resources/production
echo "├── Complete"
