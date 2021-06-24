# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:40:09 2021

@author: Amogh
"""


import pip

#check whether the package is installed if not install it.
def import_or_install(package):
    try:
        __import__(package)
        print('Package is present')
    except ImportError:
        print('Package is not Present, Trying to Install it')
        pip.main(['install', package]) 
        
import_or_install('pip')
import_or_install('sqlite3')
import_or_install('pandas')
import_or_install('matplotlib')
import_or_install('seaborn')
import_or_install('os')
import_or_install('numpy')


# importing all the necessary libraries
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# get current path of the file
path = os.getcwd()
path=os.path.dirname(os.path.abspath(__file__))

#change working directory to save the outputs
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#create aResults folder to drop all the output files
results_dir = os.path.join(path, 'Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


#connection to sqlite DataBase
con=sqlite3.connect('chinook.db')


#function to display values in the pie chart
def func(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} $)".format(pct, absolute)

#Test connection to check whether its running fine
def select_all(con):
    cur = con.cursor()
    cur.execute("SELECT * FROM tracks")

    rows = cur.fetchall()

    for row in rows:
        print(row)

# querry to run the command 
def run_only_querry(c):
    with sqlite3.connect('chinook.db') as conn:
        conn.isolation_level = None
        conn.execute(c)    

#This function run the sql querry and converts the results to pandas dataframe
def run_query(q):
    return pd.read_sql(q, con)




s1="Select * from media_types"
s2="Select * from genres"
s3="Select * from playlists"
s4="Select * from playlist_track"
s5="Select * from tracks"
s6="Select * from artists"
s7="Select * from invoices"
s8="Select * from invoice_items"
s9="Select * from albums"
s10="Select * from customers"
s11="Select * from employees"


media_types_df=run_query(s1)
media_types_df.isnull().sum()

genres_df=run_query(s2)
genres_df.isnull().sum()

run_query(s1).isnull().sum()
run_query(s2).isnull().sum()
run_query(s3).isnull().sum()
run_query(s4).isnull().sum()
run_query(s5).isnull().sum()
run_query(s6).isnull().sum()
run_query(s7).isnull().sum()
run_query(s8).isnull().sum()
run_query(s9).isnull().sum()
run_query(s10).isnull().sum()
run_query(s11).isnull().sum()



"""Q1"""


#lets look at how Genre type and Length of the track are correlated


genre_vs_length="""
SELECT 
    genres.Name, 
    tracks.Milliseconds / (60000) AS minutes 
FROM genres 
INNER JOIN tracks ON genres.GenreId=tracks.GenreId 
GROUP BY genres.Name 
ORDER BY minutes 
DESC LIMIT 10;"""

genre_vs_length_df=run_query(genre_vs_length)

#bar plot

plt.figure(figsize=(15,15))
ax=sns.barplot(x='Name', y='minutes',data=genre_vs_length_df)
#displaying the bar chart values.
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
#Rotating the x ticks labels by 90 degree
plt.xticks(rotation=45)
plt.xticks(size = 15)
#Title of the plot
plt.title("Genre vs Length of track in minutes")
# X axis Label
plt.xlabel("Genre")
#yaxis Label
plt.ylabel("Minutes")
#saving the plot to results directory
plt.savefig(results_dir+'genre_vs_minutes')
#displaying the plot
plt.show()

"""Q2"""

#Media type and its Size in MegaBytes

media_type_size='''
SELECT 
    m.Name AS Type,
    t.Bytes/1000000.0 AS Size
FROM media_types m 
INNER JOIN tracks t ON m.MediaTypeId=t.MediaTypeId 
GROUP BY Type'''

media_type_size_df=run_query(media_type_size)

plt.figure(figsize=(20,20))
ax=sns.barplot(x='Type', y='Size',data=media_type_size_df)
#displaying the bar chart values.
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=14, xytext=(0, 8),
                   textcoords='offset points')
#Rotating the x ticks labels by 90 degree
plt.xticks(rotation=45)
plt.xticks(size = 15)
#Title of the plot
plt.title("Media Type vs Size in MB")
# X axis Label
plt.xlabel("Media Type")
#yaxis Label
plt.ylabel("Size in MegaBytes")
#saving the plot to results directory
plt.savefig(results_dir+'media_type_sizes')
#displaying the plot
plt.show()

"""Q3"""

#CSV file which contains details about song, album, media type, genre, length and its price.
#This CSV file has application like in iTunes Store, where it provides information for the end users.

song_details='''
SELECT
  t.Name AS Song,
  ar.Name as ArtistName,
  a.Title as Album,
  mt.Name AS MediaType,
  g.Name AS Genre,
  Round(t.Milliseconds/ 60000.0,2) AS minutes,
  t.Unitprice as Price
FROM tracks t
LEFT JOIN media_types mt on mt.MediaTypeId = t.MediaTypeId
LEFT JOIN albums a ON a.AlbumId = t.AlbumId
LEFT JOIN artists ar ON a.ArtistId=ar.ArtistId
LEFT JOIN genres g ON g.GenreId = t.GenreId'''

# saving the dataframe to csv file

song_details_df=run_query(song_details)
song_details_df.to_csv(results_dir+'Song_Album_genre_Type_length_price_details.csv',index=False)


#Top Tracks that are purchased

"""Q4"""

top_tracks = '''
SELECT 
    t.Name trackname,
    a.Title album_title,
    ar.Name artist,
    COUNT(*) as total_purchases,
    SUM(it.UnitPrice) total_cost
FROM tracks t 
JOIN albums a on a.AlbumID = t.AlbumID
JOIN artists ar on ar.ArtistId = a.ArtistId
JOIN invoice_items it on it.TrackId = t.TrackId
GROUP BY 1
ORDER BY total_purchases desc
'''
# saving the dataframe to csv file
top_tracks_df = run_query(top_tracks)
top_tracks_df.to_csv(results_dir+'Top_tracks_purchased.csv',index=False)

#Sales by counrty

"""Q5"""
"""Sales by country"""
sales_by_country = '''
WITH country_s AS
    (
     SELECT
       CASE
           WHEN (
                 SELECT count(*)
                 FROM customers
                 where Country = c.Country
                ) = 1 THEN "Other"
           ELSE c.Country
       END AS Country,
       c.CustomerId,
       it.*
     FROM invoice_items it
     INNER JOIN invoices i ON i.InvoiceId = it.InvoiceId
     INNER JOIN customers c ON c.CustomerId = i.CustomerId
    )


SELECT
    Country,
    SUM(UnitPrice) total_sales
    FROM country_s
    GROUP BY Country
    ORDER BY total_sales DESC;
'''

sales_by_country2='''
SELECT 
    c.Country AS Country,
    Sum(i.Total) AS sales 
FROM customers c 
INNER JOIN invoices i on c.CustomerId=i.CustomerId
GROUP BY Country 
ORDER BY sales DESC LIMIT 10;'''

#Pie chart distribution of Sales by country
sales_by_country_df=run_query(sales_by_country)
sales_by_country_df2=run_query(sales_by_country2)


plt.figure(figsize=(10,10))
plt.pie('sales', labels='Country',autopct=lambda pct: func(pct, sales_by_country_df2['sales']), shadow=True, startangle=140,data=sales_by_country_df2)
plt.title("Sales By Country",y=1.07)
plt.axis('equal')
plt.savefig(results_dir+'sales_by_country')
plt.show()


"""Q6"""
# Yearly Sales
invoice_vs_sales='''
SELECT 
    i.Total AS sales, 
    i.InvoiceDate 
FROM invoices i
INNER JOIN invoice_items it ON it.InvoiceId=i.InvoiceId;'''


invoice_vs_sales_df=run_query(invoice_vs_sales)
invoice_vs_sales_df.dtypes

#converting InvoiceDate to datetime object
invoice_vs_sales_df['InvoiceDate'] = pd.to_datetime(invoice_vs_sales_df['InvoiceDate'])

#Splitting InvoiceDate into Day, Month and Year for plotting. 
invoice_vs_sales_df['Day'] = invoice_vs_sales_df['InvoiceDate'].dt.day
invoice_vs_sales_df['Month'] = invoice_vs_sales_df['InvoiceDate'].dt.month
invoice_vs_sales_df['Year'] = invoice_vs_sales_df['InvoiceDate'].dt.year


invoice_vs_sales_year=invoice_vs_sales_df[['Year','sales']]
invoice_vs_sales_year=invoice_vs_sales_year.groupby(['Year'],as_index=False).sum()
invoice_vs_sales_year['Year'] = invoice_vs_sales_year['Year'].apply(str)

#line plot of Yearly Sales

plt.figure(figsize=(10,10))
plt.plot(invoice_vs_sales_year['Year'],invoice_vs_sales_year['sales'])
plt.title('Yearly Sales')
plt.ylabel('Sales in $')
plt.xlabel('Year')
plt.grid()
plt.savefig(results_dir+"sales_vs_year.png")
plt.show()


#2009 Sales Data
invoice_vs_sales_2009_df=invoice_vs_sales_df[invoice_vs_sales_df['Year']==2009]
invoice_vs_sales_2009_df=invoice_vs_sales_2009_df.groupby(['Month','Year'],as_index=False).sum()

#Line plot of Monthly Sales in 2009



plt.figure(figsize=(5,5))
plt.plot(invoice_vs_sales_2009_df['Month'], invoice_vs_sales_2009_df['sales'])
plt.title('2009 Sales')
plt.ylabel('Sales in $')
plt.xlabel('Month')
plt.ylim([100, 500])
plt.grid()
plt.savefig(results_dir+"2009_sales.png")
plt.show()

#2010 Sales

invoice_vs_sales_2010_df=invoice_vs_sales_df[invoice_vs_sales_df['Year']==2010]
invoice_vs_sales_2010_df=invoice_vs_sales_2010_df.groupby(['Month','Year'],as_index=False).sum()

#Line plot of Monthly Sales in 2010

plt.figure(figsize=(5,5))
plt.plot(invoice_vs_sales_2010_df['Month'],invoice_vs_sales_2010_df['sales'])
plt.title('2010 Sales')
plt.ylabel('Sales in $')
plt.xlabel('Month')
plt.ylim([100, 500])
plt.grid()
plt.savefig(results_dir+"2010_sales.png")
plt.show()

#2011 Sales
invoice_vs_sales_2011_df=invoice_vs_sales_df[invoice_vs_sales_df['Year']==2011]
invoice_vs_sales_2011_df=invoice_vs_sales_2011_df.groupby(['Month','Year'],as_index=False).sum()

#Line plot of Monthly Sales in 2011
plt.figure(figsize=(5,5))
plt.plot(invoice_vs_sales_2011_df['Month'],invoice_vs_sales_2011_df['sales'])
plt.title('2011 Sales')
plt.ylabel('Sales in $')
plt.xlabel('Month')
plt.ylim([100, 500])
plt.grid()
plt.savefig(results_dir+"2011_sales.png")
plt.show()


#2012 Sales
invoice_vs_sales_2012_df=invoice_vs_sales_df[invoice_vs_sales_df['Year']==2012]
invoice_vs_sales_2012_df=invoice_vs_sales_2012_df.groupby(['Month','Year'],as_index=False).sum()

#Line plot of Monthly Sales in 2012
plt.figure(figsize=(5,5))
plt.plot(invoice_vs_sales_2012_df['Month'],invoice_vs_sales_2012_df['sales'])
plt.title('2012 Sales')
plt.ylabel('Sales in $')
plt.xlabel('Month')
plt.ylim([100, 500])
plt.grid()
plt.savefig(results_dir+"2012_sales.png")
plt.show()

#2013 Sales
invoice_vs_sales_2013_df=invoice_vs_sales_df[invoice_vs_sales_df['Year']==2013]
invoice_vs_sales_2013_df=invoice_vs_sales_2013_df.groupby(['Month','Year'],as_index=False).sum()

#Line plot of Monthly Sales in 2013
plt.figure(figsize=(5,5))
plt.plot(invoice_vs_sales_2013_df['Month'],invoice_vs_sales_2013_df['sales'])
plt.title('2013 Sales')
plt.ylabel('Sales in $')
plt.xlabel('Month')
plt.ylim([100, 500])
plt.grid()
plt.savefig(results_dir+"2013_sales.png")
plt.show()


#year month wise sales

invoice_vs_sales_month_year_df=invoice_vs_sales_df[['sales','InvoiceDate']]
invoice_vs_sales_month_year_df['month_year']=invoice_vs_sales_month_year_df['InvoiceDate'].dt.strftime('%Y-%m')
invoice_vs_sales_month_year_df=invoice_vs_sales_month_year_df.groupby(['month_year'],as_index=False).sum()
invoice_vs_sales_month_year_df['month_year'] = invoice_vs_sales_month_year_df['month_year'].apply(str)



ax = invoice_vs_sales_month_year_df.plot(x='month_year', y='sales')
plt.title('Monthly Sales')
plt.savefig(results_dir+"sales_vs_moth_year.png")
plt.show()

"""Q7"""
#Genre vs sales

genre_vs_sales="""
SELECT
     g.Name, 
     Sum(it.UnitPrice) AS Price 
FROM genres g
INNER JOIN tracks t ON g.GenreId=t.GenreId 
INNER JOIN invoice_items it ON t.TrackId=it.TrackId 
GROUP BY g.Name 
ORDER BY Price DESC LIMIT 5;"""



genre_vs_sales_df=run_query(genre_vs_sales)
    
#pie Chart of Genre vs Sales
plt.figure(figsize=(7,7))   
plt.pie('Price', labels='Name',autopct=lambda pct: func(pct, genre_vs_sales_df['Price']), shadow=True, startangle=140,data=genre_vs_sales_df)
plt.title("Top 5 Genres by Sales",y=1.07)
plt.axis('equal')
plt.savefig(results_dir+'sales_by_genres')
plt.show()



"""Q8"""

#sales by city
sales_vs_city='''
SELECT 
    i.BillingCity AS City, 
    Sum(it.UnitPrice) AS sales 
FROM invoices i
INNER JOIN invoice_items it ON i.InvoiceId=it.InvoiceId 
GROUP BY City 
ORDER BY sales DESC;'''


sales_vs_city_df=run_query(sales_vs_city).head(10)

plt.figure(figsize=(10,10))
ax=sns.barplot(x='City', y='sales',data=sales_vs_city_df)
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=12, xytext=(0, 8),
                   textcoords='offset points')
plt.xticks(rotation=45)
plt.xticks(size = 15)
plt.title("Top 10 Cities By Sales")
plt.xlabel("City")
plt.ylabel("Sales")
plt.savefig(results_dir+'top_10_city_by_sales')
plt.show()


"""Q9"""
"""
#Most purchased Track
most_pur_track='''
SELECT 
    t.Name as "Track Name", 
    COUNT(it.TrackId) AS "Number of Purchases" 
FROM tracks t, 
    invoice_items it, 
    invoices i 
WHERE it.TrackId == t.TrackId
AND it.InvoiceId == i.InvoiceId
GROUP BY t.Name
ORDER BY COUNT(it.TrackId) DESC;'''

#saving the dataframe to csv
most_pur_tracks_df=run_query(most_pur_track)
most_pur_tracks_df.to_csv(results_dir+'most_purchased_track.csv',index=False)

"""

"""Q10"""

#Which sales agent got the most sales
most_sales_by_agent='''
SELECT 
    e.FirstName || " " || e.LastName AS "Sales Agent", 
    SUM(i.Total) AS "Total Sales"
FROM employees e
INNER JOIN customers c ON c.SupportRepId = e.EmployeeId
INNER JOIN invoices i ON i.CustomerId = c.CustomerId
AND e.Title == "Sales Support Agent"
GROUP BY e.FirstName || " " || e.LastName
ORDER BY SUM(i.Total) DESC;
'''

most_sales_by_agent_df=run_query(most_sales_by_agent)


#Pie chart distribution of sales agent's performace
plt.figure(figsize=(10,10))
plt.pie('Total Sales', labels='Sales Agent',textprops={'fontsize': 14},autopct=lambda pct: func(pct, most_sales_by_agent_df['Total Sales']), shadow=True, startangle=140,data=most_sales_by_agent_df)
plt.axis('equal')
plt.savefig(results_dir+'sales_by_agents')
plt.show()

"""Q11"""
#Employee performance Year wise and comparing with each other.
employee_sales_performance = '''
WITH 
    customer_support_rep_sales AS
        (
         Select 
             i.CustomerId,
             i.InvoiceDate DateTime,
             c.SupportRepId,
             SUM(i.total) Sales_Total
         FROM invoices i
         INNER JOIN customers c ON c.CustomerId = i.CustomerId
         GROUP BY 2, 3
        )
SELECT
    e.FirstName || " " || e.LastName "Employee Name",
    csrs.DateTime,
    SUM(csrs.Sales_total) "Amount of Sales"
FROM customer_support_rep_sales csrs
INNER JOIN employees e ON csrs.SupportRepId = e.EmployeeId
GROUP BY 1, 2;
'''


#saving the results to dataframe
employee_sales_performance_df = run_query(employee_sales_performance)
#converting to datetime object
employee_sales_performance_df["DateTime"] = pd.to_datetime(employee_sales_performance_df["DateTime"])

#split year and month
employee_sales_performance_df['Year']=employee_sales_performance_df['DateTime'].dt.strftime('%Y')
employee_sales_performance_df['Month']=employee_sales_performance_df['DateTime'].dt.strftime('%m')

employee_sales_performance_df.to_csv(results_dir+'employee_sales_performance.csv',index=False)

#Grouping the sales by year and Employee Name 
employee_sales_performance_df=employee_sales_performance_df.groupby(['Employee Name','Year'],as_index=False).sum()

#Subsetting each employee for better visualization
employee_sales_performance_df_jane=employee_sales_performance_df[employee_sales_performance_df['Employee Name']=='Jane Peacock']
employee_sales_performance_df_park=employee_sales_performance_df[employee_sales_performance_df['Employee Name']=='Margaret Park']
employee_sales_performance_df_steve=employee_sales_performance_df[employee_sales_performance_df['Employee Name']=='Steve Johnson']

#x labels
labels =employee_sales_performance_df['Year']
#y values(Employee Sales)
jane = employee_sales_performance_df_jane['Amount of Sales']
park=employee_sales_performance_df_park['Amount of Sales']
steve=employee_sales_performance_df_steve['Amount of Sales']

# set Width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 10))

# Set position of bar on X axis
br1 = np.arange(len(jane))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Bar plot
plt.bar(br1, jane, color ='y', width = barWidth,
        edgecolor ='grey', label ='Jane Peacock')
plt.bar(br2, park, color ='b', width = barWidth,
        edgecolor ='grey', label ='Margaret Park')
plt.bar(br3, steve, color ='g', width = barWidth,
        edgecolor ='grey', label ='Steve Johnson')
 
# Adding Xtick and respective x and y labels.
plt.xlabel('Year', fontweight ='bold', fontsize = 15)
plt.ylabel('Sales', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(jane))],
        ['2009', '2010', '2011', '2012', '2013'])
plt.title("Employee Sales Performance")
plt.legend()
plt.savefig(results_dir+"employee_performace")
plt.show()




"""Q12"""
#Invoice details of customer,employee, country, price, trackname, genre, mediatype
invoice_details="""
SELECT 
    e.FirstName || " " || e.LastName  AS EmployeeName, 
    c.FirstName || " " || c.LastName AS CustomerName, 
    c.Country AS "Customer Country" ,
    it.UnitPrice AS Price, 
    c.City, i.InvoiceDate, 
    t.Name AS TrackName, 
    g.Name AS Genre,
    mt.Name AS MediaType
FROM employees e 
INNER JOIN customers c ON e.EmployeeId=c.SupportRepId 
INNER JOIN invoices i ON i.InvoiceId=c.CustomerId
INNER JOIN invoice_items it ON i.InvoiceId=it.InvoiceId
INNER JOIN tracks t ON t.TrackId=it.TrackId
INNER JOIN genres g ON t.GenreId=g.GenreId
INNER JOIN media_types mt ON mt.MediaTypeId=t.MediaTypeId
 """

invoice_details_df=run_query(invoice_details)
invoice_details_df.to_csv(results_dir+'Invoice details.csv',index=False)


"""13"""
view='''CREATE VIEW custom_country AS
WITH count_oth AS (
    SELECT
        COUNT(c.CustomerId),
        CASE
            WHEN COUNT(c.CustomerId) = 1 THEN "Other"
            ELSE c.Country
            END AS Country_others
    FROM customers c
    GROUP BY c.Country
    )
SELECT
    c.CustomerId,
    CASE
        WHEN c.Country NOT IN (SELECT country_others FROM count_oth) THEN "Other"
        ELSE c.Country
        END AS country_other
FROM customers c'''

#run_only_querry(view)

view_querry = "SELECT * FROM custom_country"
run_query(view_querry)


country_study = '''
SELECT
    cc.country_other AS Country,
    COUNT(cc.CustomerId) AS No_of_Customers,
    SUM(i.total) AS total_sales,
    ROUND(CAST(SUM(i.total) as float) / COUNT(DISTINCT(cc.CustomerId)),2) Avg_sales_per_cust,
    ROUND(AVG(i.total),2) Avg_order_value
FROM custom_country cc
INNER JOIN invoices i ON i.CustomerId = cc.CustomerId
GROUP BY 1
ORDER BY 
    CASE 
        WHEN country = "Other" THEN 0
        ELSE total_sales
        END
DESC LIMIT 9;'''

country_study_df=run_query(country_study)
country_study_df.to_csv(results_dir+'country_study.csv',index=False)

plt.figure(figsize=(10,10))
ax=sns.barplot(x='Country', y='No_of_Customers',data=country_study_df)
for bar in ax.patches:
    ax.annotate(int(bar.get_height()), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xticks(rotation=45)
plt.xticks(size=12)
plt.title("Country vs Customers")
plt.xlabel("Country")
plt.ylabel("No of Customers")
plt.savefig(results_dir+'country_cust')
plt.show()



#total Sales vs Country

plt.figure(figsize=(10,10))
ax=sns.barplot(x='Country', y='total_sales',data=country_study_df)
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xticks(rotation=45)
plt.xticks(size=12)
plt.title("Country vs Sales")
plt.xlabel("Country")
plt.ylabel("Sales in $")
plt.savefig(results_dir+'country_sales')
plt.show()


#avg sales by customer
plt.figure(figsize=(10,10))
ax=sns.barplot(x='Country', y='Avg_sales_per_cust',data=country_study_df)
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xticks(rotation=45)
plt.title("Country vs Avg Sales per Customer")
plt.xlabel("Country")
plt.ylabel("Sales in $")
plt.savefig(results_dir+'Avg_sales_per_Customer')
plt.show()

#avg Transaction

plt.figure(figsize=(10,10))
ax=sns.barplot(x='Country', y='Avg_order_value',data=country_study_df)
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), '.2f'), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xticks(rotation=45)
plt.title("Country vs Avg Order Value")
plt.xlabel("Country")
plt.ylabel("Sales in $")
plt.savefig(results_dir+'Avg_Order_Value')
plt.show()


"""Q14"""
#Since Prague and Czech Repulic is an emerging market, lets look at its Music Taste

czec_rep_genre = '''
WITH 
cz_sales AS (
    SELECT 
        SUM(quantity) total
    FROM invoice_items it
    INNER JOIN invoices i ON i.InvoiceId = it.invoiceId
    WHERE i.BillingCountry = 'Czech Republic')
SELECT 
    g.Name genre, 
    SUM(it.Quantity) num_sold, 
    ROUND((CAST(SUM(it.Quantity) as float) / (SELECT total FROM cz_sales)) * 100, 2) \
        percent_sold 
FROM invoice_items it
INNER JOIN invoices i ON i.InvoiceId = it.InvoiceId 
INNER JOIN tracks t ON it.TrackId = t.TrackId 
INNER JOIN genres g ON t.GenreId = g.GenreId 
WHERE i.BillingCountry = 'Czech Republic' 
GROUP BY 1 
ORDER BY 2 DESC LIMIT 6;
'''

#No of tracks sold 

my_file='Czech Republic Genre Taste'


czec_rep_genre_df=run_query(czec_rep_genre)
czec_rep_genre_df.to_csv(results_dir+'Czech_republic_genre.csv',index=False)

plt.figure(figsize=(15,15))
plt.bar(x='genre',height='num_sold',data=czec_rep_genre_df)
plt.xticks(rotation=45)
plt.xticks(fontsize=15)
plt.title("Czech Republic Genre Taste")
plt.xlabel("Genre")
plt.ylabel("Number of copies sold")
plt.savefig(results_dir+my_file)
plt.figure(figsize=(20,20))
plt.show()


"""Q15"""
#No of invoices per country
invoices_per_country='''
SELECT 
    BillingCountry AS Country, 
    count(BillingCountry) AS 'Invoices_Count'
FROM invoices
GROUP BY Country
ORDER BY Invoices_Count DESC LIMIT 5; '''


invoices_per_country_df=run_query(invoices_per_country)


plt.figure(figsize=(10,10))
ax=sns.barplot(x='Country', y='Invoices_Count',data=invoices_per_country_df)
for bar in ax.patches:
    ax.annotate(int(bar.get_height()), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.title("Top 5 Countries vs Invoice Count")
plt.xlabel("Country")
plt.ylabel("Invoice Count")
plt.savefig(results_dir+'invoices_per_country')
plt.show()

"""Q16"""
#no of customers assigned to an employee
no_of_cust_emp='''
SELECT
  e.FirstName || " " || e.LastName as Employee,
  COUNT(*) AS Customer_Count
FROM customers c
JOIN employees e ON e.EmployeeId = c.SupportRepId
GROUP BY e.EmployeeId;'''

no_of_cust_emp_df=run_query(no_of_cust_emp)

plt.figure(figsize=(10,10))
ax=sns.barplot(x='Employee', y='Customer_Count',data=no_of_cust_emp_df)
for bar in ax.patches:
    ax.annotate(int(bar.get_height()), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.title("No of Customers assigned to Sales Agent")
plt.xlabel("Sales Agent")
plt.ylabel("No of Customers")
plt.savefig(results_dir+'no_of_cust_emp')
plt.show()


"""Q17"""
#Count number of Tracks for the artist
count_songs_album_artist='''
SELECT
	t.AlbumId,
	ab.Title as AlbumName,
	COUNT(trackid) as NumberOFTracks,
	a.name as ArtistName,
    g.Name as Genre
FROM
tracks t 
INNER JOIN albums ab on t.AlbumId == ab.AlbumID 
INNER JOIN artists a on a.ArtistId == ab.ArtistId
INNER JOIN genres g on g.GenreId=t.GenreId
GROUP BY
	t.AlbumId;
    '''
    
count_songs_album_artist_df=run_query(count_songs_album_artist)
count_songs_album_artist_df.to_csv(results_dir+'Artist_Album_NoofTracks.csv',index=False)


count_songs_album_artist_df=count_songs_album_artist_df.sort_values(by='NumberOFTracks',ascending=False).head(10)

plt.figure(figsize=(10,15))
ax=sns.barplot(x='ArtistName', y='NumberOFTracks',data=count_songs_album_artist_df)
for bar in ax.patches:
    ax.annotate(int(bar.get_height()), 
                   (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.title("No of Tracks produced by Artist")
plt.xticks(rotation=45)
plt.xlabel("Artists")
plt.ylabel("No of Tracks")
plt.savefig(results_dir+'artist_tracks')
plt.show()


#czech Republic's Most purchased track
"""Q18"""
czech_rep_track='''
SELECT 
    t.name as TrackName,
    count(billingcountry) as purchase_count ,
    ar.Name as ArtistName,
    g.Name as Genre
    FROM invoice_items it
INNER JOIN invoices i on i.invoiceId = it.invoiceId 
INNER JOIN tracks t ON  t.trackId = it.trackId
INNER JOIN albums a ON a.AlbumId=t.AlbumId
INNER JOIN artists ar ON ar.ArtistId=a.ArtistId
INNER JOIN genres g on g.GenreId=t.GenreId
where i.billingcountry = "Czech Republic" 
GROUP BY it.invoiceId
ORDER BY count(t.trackid) DESC'''

czech_rep_track_df=run_query(czech_rep_track)
czech_rep_track_df.to_csv(results_dir+'Most_listned_track_CZ.csv',index=False)


"""Master DB"""
"""Q19"""

#master Database File
master_query="""
SELECT
    i.InvoiceID, 
    c.FirstName || " " || c.LastName  AS CustomerName,
    e.FirstName || " " || e.LastName  AS CustomerName,
    c.City,
    c.Phone,
    c.Email, 
    g.Name as Genre,
    mt.Name as MediaType,
    t.Name as TrackName,
    it.UnitPrice as Price,
    a.Title as Album,
    ar.Name as ArtistName
    FROM customers c
    LEFT JOIN employees e ON c.SupportRepId=e.EmployeeId
    INNER JOIN invoices i ON i.CustomerId=c.CustomerId
    INNER JOIN invoice_items it ON it.InvoiceId=i.InvoiceId
    INNER JOIN tracks t ON t.TrackId=it.TrackId
    INNER JOIN genres g On g.GenreId=t.GenreId
    INNER JOIN media_types mt ON mt.MediaTypeID=t.MediaTypeId
    INNER JOIN albums a ON t.AlbumId=a.AlbumId
    INNER JOIN artists ar ON ar.ArtistId=a.ArtistId
    group by i.InvoiceId;
"""

master_query_df=run_query(master_query)
master_query_df.to_csv(results_dir+'master_file.csv',index=False)



#top Playlisted Track
"""Q20"""
playlist_count='''
SELECT
    t.Name as TrackName,
    count(pt.TrackId) as count
FROM tracks t
INNER JOIN playlist_track pt ON t.TrackId= pt.TrackId
GROUP BY TrackName
ORDER BY count DESC;
'''

playlist_count_df=run_query(playlist_count)

playlist_count_df.to_csv(results_dir+'Top Playlisted Track.csv',index=False)


#Emplyee Performace with hire date
employee_sales_performance = '''
WITH 
    sales_rep AS
        (
         SELECT 
             i.CustomerId,
             c.SupportRepId,
             SUM(i.Total) Sales
         FROM invoices i
         INNER JOIN customers c ON c.CustomerId = i.CustomerId
         GROUP BY 2
        )
SELECT
    e.FirstName || " " || e.LastName "Employee Name",
    e.HireDate "Hire Date",
    SUM(sr.Sales) "Sales (in $)"
FROM sales_rep sr
INNER JOIN employees e ON sr.SupportRepId = e.EmployeeId
GROUP BY 1;
'''
emp_df=run_query(employee_sales_performance)