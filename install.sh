echo "Installing necessary packages:"
sudo apt install nodejs npm python3-pip

echo "Switching to Emo-Sense directory"
cd Emo-Sense

echo "Installing packages for backend and running server"
cd backend
npm install
node server.js &

echo "Installing packages for frontend and running client"
cd ../frontend
npm install
npm start &

echo "Installing packages for Flask and running Flask API"
cd ../Flask\ Testing
pip3 install -r requirements.txt
python main.py &
