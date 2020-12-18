
# The AWS Ecosystem

## Introduction

In this lesson, we'll get set up to use **_Amazon Web Services_**, and then get to know our way around the platform before digging into AWS SageMaker in the next lesson. 

<img src='images/awscloud.svg'>


## Objectives 

- Set up an AWS account and explore the Amazon Resource Center 
- Explain what the "regions" are in AWS and why it is important to choose the right one 


## Getting Started

Before we can begin exploring everything AWS has to offer, we'll need to create an account on the platform. To do this, start by following this link to [Amazon Web Services](https://aws.amazon.com/). While you're there, you may want to take the time to bookmark it -- chances are this is a website you'll use frequently in your career as a Data Scientist!

### Will This Cost Money?

Although you will need a credit card to register for AWS, working through this section will not cost any money. AWS provides a free tier for learning and prototyping on the platform -- this is the tier we'll use for everything going forward. As long as you correctly register for the free tier, this will not cost you any money. 

### Register Your Email

Begin by clicking the "Sign Up" button in the top right-hand corner of the page. 

<img src='images/aws-1.png'>

Next, create an account by adding your email and password. You'll also need to set an **_AWS Account Name_**. 

<img src='images/aws-2.png'>

On the next screen, enter your contact information. **_Make sure you set your account type to 'Personal'!_** 

<img src='images/aws-3.png'>

This next page is especially important -- be sure to select the **_Basic Plan_**! As a reminder, you will be asked to enter a credit card number during the next few steps. Although we will only be making use of the free tier of services for AWS, be aware that you will still need to enter a credit card number in order to complete the registration process. 

<img src='images/aws-4.png'>

Now that you're all signed up, click the "Sign in to the Console" button to actually enter the AWS Console. 

<img src='images/aws-5.png'>

Alright, you've now created an AWS Account! Let's take a look around. 

## The AWS Console

Now that you're signed in, you'll see the **_AWS Console_**. This is your "home screen" for AWS -- it allows you to quickly navigate through the thousands of services offered on AWS to find what you need. The easiest way to find what you need is the "Find Services" search bar at the top of the body of the page. 

<img src='images/aws-6.png'>

You can also click the "See All Services" dropdown to see a full list of services you can use in AWS. There are **a ton** of services, but don't let yourself get overwhelmed -- you'll probably never end up using the vast majority of these, as only a few apply to the work of a data scientist. 

## Use Cases for Data Scientists

We've now created an account for AWS, so that we can take advantage of the Cloud. As data scientists, we'll find that a cloud computing service like AWS is very helpful in a number of ways. Aside from productionizing the model as a whole, the most important thing the cloud enables data scientists to do is to train much, much larger models by distributing training across entire clusters of servers. Without cloud computing, it would be impossible to train some of the larger deep learning models that exist today. The ability to distribute training of a neural network across a GPU allowed AI researchers to create massive models in a reasonable amount of time by creating a server cluster full of hundreds of GPUs. While this works, building a server like this is cost prohibitive to all but major companies and universities. Thankfully, services like AWS allow us to rent time on these servers per minute, making distributed training available for anybody at extremely cheap prices, paying only for what we use. AWS provides other great uses for data scientists beyond speedy training times -- it also plays a major part with databases. In your job as a data scientist, the databases you connect to in order to get your data will almost certainly be stored on AWS, or a competitor cloud platform. AWS servers also allow for companies to make use of big data frameworks such as Hadoop or Spark across a cluster of servers. 

## Using the Amazon Resource Center

As platforms go, you won't find many with more options than AWS. It has an amazing amount of offerings, with more getting added all the time. While AWS is great for basic use cases like hosting a server or a website, it also has all kinds of different offerings in areas such as Databases, Machine Learning, Data Analytics and other areas useful to Data Scientists. It's not possible for us to cover how to use every service in AWS in this section -- but luckily, we don't need to, because Amazon already has! The [Getting Started Resource Center](https://aws.amazon.com/getting-started/) contains a ton of awesome tutorials, demonstrations, and sample projects for just about everything you would ever want to know about any service on AWS. We **_strongly recommend_** bookmarking this page, as the tutorials they offer are very high quality, and free!

<img src='images/aws-7.png'>


## A Note On Regions

Before we move onto digging into **_AWS SageMaker_** in the next lesson, it's worth taking a moment to explain "Regions" and what they have to do with AWS. AWS has data centers all over the world, and they are **not** interchangeable when it comes to your projects. Click on the "Region" tab in the top right corner of the navigation bar, and you should see a dropdown of all the different data centers you can choose from. It is **_very important_** that you always choose the same region to connect to with your projects. Each region is its own unique data center, and anything you do on in that region is only in that region. One of the most common mistakes newcomers to AWS make is thinking they've lost their project because they are connected to a different data center and don't realize it. We'll remind you of this again later, but it can't hurt to say it twice: always make sure you're connected to the correct data center! This goes doubly for when you're creating a new project. 

## Summary

In this lesson, we signed up for Amazon Web Services and explored some of the different options on the platform. We also learned about where AWS fits in the data science process. 
