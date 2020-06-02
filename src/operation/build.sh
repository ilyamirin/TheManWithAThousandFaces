cd $(dirname $0)/../..

echo "Building App..."
docker build -t fac-app -f App.Dockerfile .
echo "├── Complete"

echo "Building UI..."
docker build --build-arg app_api_url=${REACT_APP_API_URL} -t fac-ui -f UI.Dockerfile .
echo "├── Complete"

echo "Building train budget model job..."
docker build --build-arg train_script='train_budget_model' -t fac-train-budget-model-job -f TrainModel.Dockerfile .
echo "├── Complete"

echo "Building train turnover model job..."
docker build --build-arg train_script='train_turnover_model' -t fac-train-turnover-model-job -f TrainModel.Dockerfile .
echo "├── Complete"
