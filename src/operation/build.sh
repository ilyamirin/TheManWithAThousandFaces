cd $(dirname $0)/../..

echo "Building App..."
docker build -t fac-app -f App.Dockerfile .
echo "├── Complete"

echo "Building UI..."
docker build -t fac-ui -f UI.Dockerfile .
echo "├── Complete"
