# HackathonML

[Harsh Modi](https://github.com/HarshModi2005), [Yugam Bhatt](https://github.com/Y4NK33420) and Kumar Vedant made this project as a part of the Hackathon in the *Annual TechFest of IIT Ropar - Advitiya*. Our problem statement was to tackle cyber security issues using AI/ML. We made 5 sub-projects.



# DeepFake Detector
 This model recognized deepfakes from real images by an accuracy of 94%. This can be embedded in social media websites to filter out degrading content

# Troll Police
 This scrapes the Youtube comments of a channel and runs sentiment analysis across multiple videos of the channel to identify top trolls. This can be extended to reduce negative content from all social media.

# Bot Defender
 This used the anomaly detection algorithm and historical Gaussian distribution to detect bot activity on a website. This can not only increase the security of a website but also can be used to anticipate server demands to prevent server crashes

# Malware Detector
 This identified the malware on the basis of the metadata of a file. This can be embedded in current systems to block malicious software.

# PhishingLinkGuard
 This determined the probability of a link being a phishing link just on the basis of the url. This can be embedded in the web system to block malicious websites.
# For more details
Refer to the following [ppt](https://docs.google.com/presentation/d/1tbphcocAcrDn14xjGYhzZrZYGPAx7-4mApWFbT1oiQk/edit?usp=sharing) we made for our project 
# Demo
clone the repo in your desired directory

    git clone https://github.com/HarshModi2005/HackathonML
    
   install the required libraries
   

    pip install -r requirements.txt
   
  ## for deepfake detector
  open the demo.py file from the DeepFake folder add in your file path and run it.
## for malware detector
open the infer.py file from the Malware folder add your file path in the file_path variable and run it.

## for troll police 
get your youtube comments v3 api key from [google developer dashboard](https://developers.google.com/)
run the yt_comments.py file and add the channel name when prompted.
  



