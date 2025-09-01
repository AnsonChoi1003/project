#As a Life expectancy research team, we will join as a public health strategy team member to collaborate with medical professionals and hospital authority to develop a system to collect data
#by providing census, health surveys, death registration and life table from their data entering that can let us
#to analyse the different relationship between different aspects and the trend of life expectancy.
import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk 
import csv
import os

#Recreate and collect data into csv file
def save_data():
    life_expectancy_type = life_expectancy_var.get() 
    year_of_death = entry_year_of_death.get()
    region = entry_region.get()
    income_group = income_group_var.get() 
    country = entry_country.get()
    sex = sex_var.get()  
    lifetime = entry_lifetime.get()

    with open('life_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([life_expectancy_type, year_of_death, region, income_group, country, sex, lifetime])
    messagebox.showinfo("Success", "Data saved successfully!")
    clear_data()  

def clear_data():
    life_expectancy_var.set("")  
    entry_year_of_death.delete(0, 'end')
    entry_region.delete(0, 'end')
    income_group_var.set("") 
    entry_country.delete(0, 'end')
    entry_lifetime.delete(0, 'end')
    sex_var.set("")  
    messagebox.showinfo("Input Cleared", "Input cleared!")

#Can show the data in the text box which entered
def show_data():
    if os.path.exists('life_data.csv'):
        with open('life_data.csv', 'r') as file:
            data = file.read()

        data_window = tk.Toplevel(window)
        data_window.title("Collected Data")
        text_area = scrolledtext.ScrolledText(data_window, width=80, height=20)
        text_area.pack(padx=10, pady=10)
        text_area.insert(tk.END, data)
        text_area.config(state=tk.DISABLED)  
    else:
        messagebox.showwarning("No Data", "No data has been collected yet.")

window = tk.Tk()
window.title("Life Expectancy Data Record System")
window.configure(background='yellow')

frame = tk.Frame(window, bg='yellow', padx=10, pady=10)
frame.pack()

#Choose the life expectancy type
tk.Label(frame, text="Life Expectancy Type:", bg='yellow', fg='black').grid(row=0, column=0, padx=10, pady=5)
life_expectancy_var = tk.StringVar()
life_expectancy_dropdown = ttk.Combobox(frame, textvariable=life_expectancy_var)
life_expectancy_dropdown['values'] = (
    "Life expectancy at birth (years)", 
    "Healthy life expectancy (HALE) at birth (years)", 
    "Life expectancy at age 60 (years)"
)
life_expectancy_dropdown.current(0)  
life_expectancy_dropdown.grid(row=0, column=1, padx=10, pady=5)

#Enter the year of death of the patient or the people
tk.Label(frame, text="Year of Death:", bg='yellow', fg='black').grid(row=1, column=0, padx=10, pady=5)
entry_year_of_death = tk.Entry(frame)
entry_year_of_death.grid(row=1, column=1, padx=10, pady=5)

#Enter the region of the patient or the people
tk.Label(frame, text="Region:", bg='yellow', fg='black').grid(row=2, column=0, padx=10, pady=5)
entry_region = tk.Entry(frame)
entry_region.grid(row=2, column=1, padx=10, pady=5)

#Choose the income group of the patient or the people
tk.Label(frame, text="Income Group:", bg='yellow', fg='black').grid(row=3, column=0, padx=10, pady=5)
income_group_var = tk.StringVar()
income_group_dropdown = ttk.Combobox(frame, textvariable=income_group_var)
income_group_dropdown['values'] = ("High_income", "Upper_middle_income", "Lower_middle_income", "Low_income")
income_group_dropdown.current(0)  
income_group_dropdown.grid(row=3, column=1, padx=10, pady=5)

#Enetr the country of the patient or the people
tk.Label(frame, text="Country:", bg='yellow', fg='black').grid(row=4, column=0, padx=10, pady=5)
entry_country = tk.Entry(frame)
entry_country.grid(row=4, column=1, padx=10, pady=5)

#Choose the sex of the patient or the people
tk.Label(frame, text="Sex:", bg='yellow', fg='black').grid(row=5, column=0, padx=10, pady=5)

sex_var = tk.StringVar()
sex_var.set("Male") 

sex_frame = tk.Frame(frame, bg='yellow')
sex_frame.grid(row=5, column=1, columnspan=2)

radio_male = tk.Radiobutton(sex_frame, text="Male", variable=sex_var, value="Male", bg='yellow', fg='black')
radio_female = tk.Radiobutton(sex_frame, text="Female", variable=sex_var, value="Female", bg='yellow', fg='black')
radio_both = tk.Radiobutton(sex_frame, text="Both sexes", variable=sex_var, value="Both sexes", bg='yellow', fg='black')

radio_male.pack(side=tk.LEFT, padx=5) 
radio_female.pack(side=tk.LEFT, padx=5)
radio_both.pack(side=tk.LEFT, padx=5)

#Enetr the lifetime of the patient or the people
tk.Label(frame, text="Lifetime (yr):", bg='yellow', fg='black').grid(row=6, column=0, padx=10, pady=5)
entry_lifetime = tk.Entry(frame)
entry_lifetime.grid(row=6, column=1, padx=10, pady=5)

#Save the data
save_button = tk.Button(frame, text="Save", command=save_data, fg='black')
save_button.grid(row=7, column=0, columnspan=1, padx=10, pady=5)

#Clear entering data
clear_button = tk.Button(frame, text="Clear", command=clear_data, fg='black')
clear_button.grid(row=7, column=1, columnspan=2, padx=10, pady=5)

#Show the data
show_button = tk.Button(frame, text="Show Data", command=show_data, fg='black')
show_button.grid(row=8, column=0, columnspan=2, padx=10, pady=5)

window.mainloop()
