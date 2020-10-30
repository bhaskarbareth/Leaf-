'''
# importing required modules 
import requests, json 

# enter your api key here 
api_key = 'AIzaSyA4auJIW8zOoFNQ3boj_2QH-LmbafjLnrM'

# url variable store url 
#url = 'https://maps.googleapis.com/maps/api/geocode/json?'
url="https://maps.googleapis.com/maps/api/geocode/json?address=1600+Amphitheatre+Parkway,+Mountain+View,+CA&key=AIzaSyA4auJIW8zOoFNQ3boj_2QH-LmbafjLnrM"

# take place as input 
place = input() 

# get method of requests module 
# return response object 
res_ob = requests.get(url + 'address =' +place + '&key =' + api_key) 
#res_ob = requests.get(url)
print(res_ob)

# json method of response object 
# convert json format data 
# into python format data. 
x = res_ob.json() 

# print the vale of x 
print(x) '''

import geocoder
g = geocoder.ip('me')
print(g.latlng)
print(g)
print(g[0])


abc=str(g[0])
xyz=abc.split(', ')
print(xyz)
print(xyz[0][1:])
print(xyz[1])


print(g.latlng[0])
print(g.latlng[1])

from  geopy.geocoders import Nominatim
geolocator = Nominatim()
city ="Mysore"
country ="India"
loc = geolocator.geocode(city+','+ country)
print("latitude is :-" ,loc.latitude,"\nlongtitude is:-" ,loc.longitude)
