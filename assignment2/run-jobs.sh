rsync -vah ./* abounasm@$1.computecanada.ca:~/assignment2
ssh -t abounasm@$1.computecanada.ca "cd ~/assignment2; python job-generator.py"
