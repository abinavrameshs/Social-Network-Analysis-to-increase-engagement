# Social-Network-Analysis-to-increase-engagement
Social Network Analysis to increase engagement

# Business Scenario : 
This project is aimed at increasing engagement on National Geographic’s Instagram page (Instagram: natgeo). The final outcome is to advise Natgeo regarding the type of content they should post more and less of – i.e., what types of images increase engagement? What types decrease engagement?

# Methodology : 

1.	Scrape 500-1000 images from the natgeo Instagram page using scrapers available online. Along with the images, scrape captions, the number of likes and the number of comments for each post.
2.	Image labels for 1500 pictures were collected using Google Cloud vision API.
3.	Create a metric (score) for engagement by using a weighted sum of #_likes and #_comments. Be sure to normalize #_likes and #_comments. Now create an engagement score = .4*#_likes (normalized) + .6*#_comments (normalized).
4.	Build a model to predict engagement with
a.	Image labels (text) as predictors.
b.	using captions to predict the same?
c.	Using both image labels and Captions as predictors
5.	Perform topic modeling (LDA) on the original image labels and check engagement score per topic
6.	Recommendations to NatGeo


Please refer to the code and document attached in the repository for more details
