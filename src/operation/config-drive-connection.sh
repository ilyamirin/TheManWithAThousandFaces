if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "Configuring connection to Google Drive..."
    echo "Please follow the link and log in as keteride@gmail.com"

    rclone config create KeterideDrive drive config_is_local false

    echo "├── Complete"
fi
