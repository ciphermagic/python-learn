from selenium import webdriver
import os

chromedriver = "C:\\Users\\cipher\\AppData\\Local\\Google\\Chrome\\Application\\chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chromedriver
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])
driver = webdriver.Chrome(chromedriver, chrome_options=options)

name = "********"
password = "********"
driver.get("https://passport.jd.com/new/login.aspx")
driver.find_element_by_class_name("login-tab-r").click()
elem_account = driver.find_element_by_name("loginname")
elem_password = driver.find_element_by_name("nloginpwd")
elem_account.clear()
elem_password.clear()
elem_account.send_keys(name)
elem_password.send_keys(password)
driver.find_element_by_id("loginsubmit").click()
