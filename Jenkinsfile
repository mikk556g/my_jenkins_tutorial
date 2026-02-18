pipeline {
    agent any  // run on the Jenkins server container

    environment {
        IMAGE_NAME = "mlops_project_image"
        COMMIT_HASH = "${GIT_COMMIT.take(7)}"
        MLFLOW_TRACKING_URI = "file:///var/jenkins_home/mlruns"  // local MLflow logs
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/yourusername/yourrepo.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m pip install --upgrade pip'
                sh 'pip3 install -r requirements.txt'
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh 'pytest'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:${COMMIT_HASH} ."
            }
        }

        stage('Train Model') {
            steps {
                sh "python3 train.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            }
        }

        stage('Evaluate Model') {
            steps {
                sh "python3 evaluate.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            }
        }

        stage('Register Model') {
            steps {
                sh "python3 register_model.py --mlflow-uri ${MLFLOW_TRACKING_URI}"
            }
        }

        stage('Push Docker Image') {
            steps {
                echo "Optional: push image to local or remote registry"
                // sh "docker tag ${IMAGE_NAME}:${COMMIT_HASH} myregistry/${IMAGE_NAME}:${COMMIT_HASH}"
                // sh "docker push myregistry/${IMAGE_NAME}:${COMMIT_HASH}"
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/model_cards/*.md', allowEmptyArchive: true
            echo 'Pipeline finished!'
        }

        success {
            echo 'Pipeline succeeded!'
        }

        failure {
            echo 'Pipeline failed. Check logs!'
        }
    }
}
