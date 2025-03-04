# Trace Master: Real-Time Facial Recognition and Case Management for Missing Person Recovery 
Overview 
Trace Master is an AI-driven web application designed to assist in identifying and reuniting 
missing persons for their families. It leverages facial recognition technology to match 
reported missing individuals with found or suspected cases. 
Github Link : https://github.com/Ramtejapendikatala/Trace-Master- 
Features 
 Login Page: Users enter their email and password to access their account. 
 OTP-Based Authentication: Secure user authentication via OTP verification. 
 Raise Complaints: Users can report missing persons with images and details. 
 Report Suspected Cases: Allows users to report found individuals for identification. 
 Facial Recognition Matching: AI-powered image comparison to detect missing 
persons. 
 Automated Case Resolution: Matches found cases and updates their status. 
 Real-Time Notifications: Alerts users when a match is found. 
 Admin Dashboard: Displays user statistics, case data, and graphical representations. 
 Secure Data Storage: Uses Firebase authentication and CSV-based case 
management. 
Technologies Used 
 Frontend: HTML, CSS, JavaScript 
 Backend: Python, Flask 
 Database: CSV files for case management 
 AI Model: InsightFace, Arch-Face for facial recognition 
 Authentication: Firebase 
File Structure 
 templates/ 
o home.html: Landing page 
o login.html: User login page 
o register.html: User registration page 
o dashboard.html: User dashboard 
o admindash.html: Admin dashboard 
 static/ 
o css/: Stylesheets 
o images/: UI images 
o complaint_data/: Storage for complaint images 
o suspected_data/: Storage for suspected person images 
o found_data/: Storage for confirmed matches 
 CSV Files 
o complaints.csv: Stores complaint records 
o suspected.csv: Stores suspected person records 
o found.csv: Stores resolved cases 
o register.csv: Stores registered user details 
o login.csv: Stores user credentials 
o notification.csv: Stores user notifications 
o otpverification.csv: Stores OTP verification records 
 Main Application 
o app.py: Flask backend application 
o tracemaster-4ae93-firebase-adminsdk.json: Firebase configuration 
Usage 
1. Navigate to the login page and enter credentials. 
2. Register for an account and complete OTP verification. 
3. Access the dashboard to submit complaints or suspected cases. 
4. Track case progress and receive real-time notifications. 
5. Admin users can monitor case statistics and manage complaints. 
Setup Instructions 
1. Install Dependencies - pip install -r requirements.txt 
2. Set Up Firebase - Add the tracemaster-4ae93-firebase-adminsdk.json file to the project root. 
3. Run the Application - python app.py 
4. Access the Web Application - Open http://127.0.0.1:5000/ in your browser. 
Future Enhancements 
 Multi-Platform Support: Develop a mobile app for broader accessibility. 
 AWS Deployment: Deploy the project on AWS using Elastic Beanstalk or EC2, 
ensuring high availability, auto-scaling, and load balancing. 
Author - Ramateja Pendikatla
