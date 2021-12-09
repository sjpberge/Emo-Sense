echo "Installing necessary packages:"
curl -fsSL https://deb.nodesource.com/setup_17.x | sudo -E bash -
sudo apt install -y nodejs
sudo apt install python3-pip

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
