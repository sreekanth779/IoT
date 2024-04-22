Django Motion Classification Project
Welcome to the Django Motion Classification project! This project leverages Django, Djongo, and Altair to classify motion using Raspberry Pi Pico accelerometers and GPS sensors.

Installation
To set up the project environment, follow these steps:

Create and Activate Virtual Environment: We recommend creating a new virtual environment using Conda with Python 3.9. You can create a Conda environment using the following command:   

1.conda create --name motion-classification python=3.9

Then, activate the environment:
2.conda activate motion-classification
Install Dependencies: Install the required dependencies listed in the requirements.txt file:

3.pip install -r requirements.txt

Database Configuration: Configure your database settings in the settings.py file. This project uses Djongo, which allows you to use MongoDB as the database backend for Django. Ensure that your MongoDB server is running and accessible.
Migrate Database: Apply migrations to create database schema:

4.python manage.py migrate

Run Development Server: Start the development server:

5.python manage.py runserver


Usage

1.Login to Dashboard: Access the website and create an account to login to the dashboard.

2.Navigate Through Pages: Explore the different pages of the website, including the dashboard, analytics, and prediction tables.

3.Enjoy the Website: Feel free to navigate between the pages and interact with the features provided.
