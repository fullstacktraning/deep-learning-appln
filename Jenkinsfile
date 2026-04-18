pipeline {
    agent any

    stages {

        stage('Clone Code') {
            steps {
                git 'https://github.com/fullstacktraning/deep-learning-appln.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t pneumonia-app .'
            }
        }

        stage('Stop Old Container') {
            steps {
                sh 'docker rm -f pneumonia-container || true'
            }
        }

        stage('Run Container') {
            steps {
                sh 'docker run -d -p 8000:8000 --name pneumonia-container pneumonia-app'
            }
        }
    }
}