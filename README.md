1- Change project name to your project name at start in m1.py file

2- Add service account key in the same directory and name it as key.json

    https://console.cloud.google.com/apis/credentials/serviceaccountkey link to generate json key. Make sure project name is same as mentioned in m1.py file

3- First run command 
    'gcloud init' to set up project
    
4- Then run command 
     'gcloud app deploy' to deploy to app engine in same directory.
     
After this it will deploy on google app engine

You can visit site using 'gcloud app browse'
     

