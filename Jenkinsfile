

def sendFailEmail(){
    // modify fail email recipient
    def failRecipient = "lanxin.chen@biomind.ai"

    def subject = "FAILURE: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'"
    def summary = "${subject} (${env.BUILD_URL})"
    def details = """
    FAILURE: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':
    Check console output at "${env.JOB_NAME} [${env.BUILD_NUMBER}] (${env.BUILD_URL})"
    """
    emailext(
        to: failRecipient,
        recipientProviders: [culprits()],
        subject: subject,
        body: details
    )
}

def notifyBuild(String buildStatus = 'STARTED',stageName) {

    // modify build status email recipient
    def buildStatusRecipient = "lanxin.chen@biomind.ai"

    // default stauts
    buildStatus = buildStatus ?: 'SUCCESSFUL'

    def subject = "${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'"
    def summary = "${subject} (${env.BUILD_URL})" 
    def details = """
    ${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':
    Stage: ${stageName}
    Check console output at "${env.JOB_NAME} [${env.BUILD_NUMBER}] (${env.BUILD_URL})"
    """
    emailext (
        subject: subject,
        body: details,
        to: buildStatusRecipient
        )
}


pipeline {
    agent {
        label 'GPU-Monet'
    }
    options {
        timeout(time: 1, unit: 'HOURS') 
    }
    environment {
        GIT_URL = "git@skymed.ai:AI-Platform/TensorGraph.git"
    }
    
    stages {
        stage('Clone & update repos') {
            steps { 
                script{
                    stageName = 'Clone & update repos'
                    try {
                        // notifyBuild('STARTED',stageName)
                        checkout changelog: false, poll: false, scm: [$class: 'GitSCM', branches: [[name: '${ghprbActualCommit}']], doGenerateSubmoduleConfigurations: false, extensions: [], submoduleCfg: [], userRemoteConfigs: [[credentialsId: '5744d880-cfb2-4925-b24a-8d5812f7dcad', name: 'tensorgraph', refspec: '+refs/pull/*/head:refs/remotes/origin/pr/*', url: 'git@skymed.ai:AI-Platform/TensorGraph.git']]]
                        //git branch: 'develop', credentialsId: '5744d880-cfb2-4925-b24a-8d5812f7dcad', url: GIT_URL
                    } catch (e) {
                        currentBuild.result = "FAILED"
                        notifyBuild(currentBuild.result, stageName)
                        throw e
                    }    
                // finally {
                //     notifyBuild(currentBuild.result, stageName)
                // }
                } 
            }    
        }

        stage('Parallel running'){
            parallel {
                stage('SonarQube'){
                    stages{
                        stage('Static Code Analysis') {
                            steps{
                                // requires SonarQube Scanner
                                withSonarQubeEnv('sonar.skymed.ai') {
                                sh '''
                                    /opt/sonar-scanner/bin/sonar-scanner -X \
                                    -Dsonar.projectKey=tensorgraph \
                                    -Dsonar.sources=. \
                                    -Dsonar.host.url=https://sonar.skymed.ai \
                                    -Dsonar.login=b6252cd52eb8c205c39c3900a5364548b8c4502e
                                '''
                                }
                            }
                        }

                        stage('Code Quality Gate') {
                            options {
                                timeout(time: 1, unit: 'HOURS')
                            }
                            steps {
                                script{
                                    def qualitygate = waitForQualityGate()
                                    def failmsg = """
                                                    Pipeline aborted due to
                                                    quality gate failure:
                                                    ${qualitygate.status}
                                                    Access to SonarQube dashboard
                                                    (https://sonar.skymed.ai/dashboard?id=tensorgraph)
                                                    to understand the warnings.
                                                  """
                                    if (qualitygate.status != "OK") {
                                        echo failmsg
                                        currentBuild.result = "FAILED"
                                        stageName = 'Code Quality Gate'
                                        notifyBuild(currentBuild.result, stageName)
                                        error "Pipeline aborted due to quality gate failure: ${qualitygate.status}"
                                    }
                                }
                            }
                        }
                    }
                }

                stage('Workflow'){
                    stages{
                        
                        stage('environment setup') {
                            steps {
                                script {
                                    stageName = 'environment setup'
                                    try {
                                        //sh 'apt-get update && apt-get install -y python-pip python3-pip'
                                        //sh 'pip install --no-cache-dir --only-binary=numpy,scipy numpy nose scipy sklearn --user'
                                        //sh 'pip install --no-cache-dir tensorflow --user'
                                        //sh 'pip3 install --no-cache-dir --only-binary=numpy,scipy numpy nose scipy sklearn --user'
                                        //sh 'pip3 install --no-cache-dir tensorflow --user'
                                        //sh "pip3 install -U pytest pytest-cov pytest-metadata pytest-qt PyQt5 pytest-xvfb pytest-remotedata>=0.3.1 --user"
                                        //sh "pip install -U pytest pytest-cov pytest-metadata pytest-qt PyQt5 pytest-xvfb pytest-remotedata>=0.3.1 --user"
                                        //sh 'python -m pytest --cov=tensorgraph test/'
                                        sh 'mkdir -p test-report'
                                        //sh 'python3 -m pytest --cov=tensorgraph --cov-report=xml:test-report/coverage.xml test/'
                                    } catch (e) {
                                        currentBuild.result = "FAILED"
                                        notifyBuild(currentBuild.result,stageName)
                                        throw e
                                    } 
                                }
                            }
                        }
                        stage('Run python3 pytest') {
                            steps {
                                script {
                                    stageName = 'Run python3 pytest'
                                    try {
                                        //sh 'apt-get update && apt-get install -y python-pip python3-pip'
                                        //sh 'pip install --no-cache-dir --only-binary=numpy,scipy numpy nose scipy pytest sklearn --user'
                                        //sh 'pip install --no-cache-dir tensorflow --user'
                                        //sh 'pip3 install --no-cache-dir --only-binary=numpy,scipy numpy nose scipy pytest sklearn --user'
                                        //sh 'pip3 install --no-cache-dir tensorflow --user'
                                        //sh "pip3 install -U pytest-cov pytest-metadata pytest-qt PyQt5 pytest-xvfb pytest-remotedata>=0.3.1 --user"
                                        //sh "pip install -U pytest pytest-cov pytest-metadata pytest-qt PyQt5 pytest-xvfb pytest-remotedata>=0.3.1 --user"
                                        //sh 'python -m pytest --cov=tensorgraph test/'
                                        //sh 'mkdir -p test-report'
                                        sh 'python3 -m pytest --cov=tensorgraph --cov-report=xml:test-report/coverage.xml test/'
                                        
                                    } catch (e) {
                                        currentBuild.result = "FAILED"
                                        notifyBuild(currentBuild.result,stageName)
                                        throw e
                                    } 
                                }
                            }
                        }
                        stage('Run python2 pytest') {
                            steps {
                                script {
                                    stageName = 'Run python2 pytest'
                                    try {
                                        //sh 'apt-get update && apt-get install -y python-pip python3-pip'
                                        //sh 'pip install --no-cache-dir --only-binary=numpy,scipy numpy nose scipy pytest sklearn --user'
                                        //sh 'pip install --no-cache-dir tensorflow --user'
                                        //sh 'pip3 install --no-cache-dir --only-binary=numpy,scipy numpy nose scipy pytest sklearn --user'
                                        //sh 'pip3 install --no-cache-dir tensorflow --user'
                                        //sh "pip3 install -U pytest-cov pytest-metadata pytest-qt PyQt5 pytest-xvfb pytest-remotedata>=0.3.1 --user"
                                        //sh "pip install -U pytest pytest-cov pytest-metadata pytest-qt PyQt5 pytest-xvfb pytest-remotedata>=0.3.1 --user"
                                        sh 'python -m pytest test/'
                                        //sh 'mkdir -p test-report'
                                        //sh 'python3 -m pytest --cov=tensorgraph --cov-report=xml:test-report/coverage.xml test/'
                                        
                                    } catch (e) {
                                        currentBuild.result = "FAILED"
                                        notifyBuild(currentBuild.result,stageName)
                                        throw e
                                    } 
                                }
                            }
                        }
                    }
                }
            }
        }   
    }

    post {
        success {
            echo 'I succeeeded!'
        }
        failure {
            echo 'I failed :('
            sendFailEmail()
        }
        unstable {
            echo 'I am unstable :/'
        }
    }
}
