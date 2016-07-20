#!/bin/sh
############################################

#echo "Performing Aptitude Update... \n"
#sudo apt-get update

#echo "Performing any upgrades (just in case)... \n"
#sudo apt-get upgrade

echo "Ok, lets get to the requirements, bear with me... \n"
sudo apt-get install python-pip npm nodejs -y

echo "I'm too lazy to check if all went well, so lets move on..."
sleep 2

# DOWNLOAD/INSTALL FOREVER TO KEEP PICAST RUNNING FOREVER... HAHA?
cd ~
sudo npm config set registry http://registry.npmjs.org/
sudo npm install -g forever
sudo npm install -g n
sudo n 6.2.2
sudo ln -sf /usr/local/n/versions/node/6.2.2/bin/node /usr/bin/node
sudo npm install -g forever-monitor

echo "Goodbye from Semaphore Installer!"
sleep 2
