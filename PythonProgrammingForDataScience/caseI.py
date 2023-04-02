
# Python Alıştırmalar

# GÖREV 1:
x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)


l = [1, 2, 3, 4,"String",3.2, False]
type(l)

d = {"Name": "Jake",
     "Age": [27,56],
     "Adress": "Downtown"}
type(d)

t = ("Machine Learning", "Data Science")
type(t)

s = {"Python", "Machine Learning", "Data Science","Python"}
type(s)

# GÖREV 2:
text = "The goal is to turn data into information, and information into insight."
text.upper().replace(","," ").replace("."," ").split()


# GÖREV 3:
lst = ["D","A","T","A","S","C","I","E","N","C","E"]
len(lst)
lst[0]
lst[10]
data_list = lst[0:4]
data_list

lst.pop(8)
lst

lst.append(101)
lst

lst.insert(8, "N")
lst


# GÖREV 4:
dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}

dict.keys()
dict.values()
dict["Daisy"][1] = 14
dict

dict.update({"Ahmet": ["Turkey", 24]})
dict

dict.pop("Antonio")
dict

# GÖREV 5:

l = [2,13,18,93,22]

def func(list):

    çift_list = []
    tek_list = []

    for i in list:
        if i % 2 == 0:
            çift_list.append(i)
        else:
            tek_list.append(i)

    return çift_list, tek_list


çift,tek = func(l)


# GÖREV 6:

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]
for i,x in enumerate(ogrenciler):
    if i<3:
        i += 1
        print("Mühendislik Fakültesi",i,". öğrenci: ",x)
    else:
        i -= 2
        print("Tıp Fakültesi",i,". öğrenci: ",x)

# GÖREV 7:

ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
  print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")



# GÖREV 8

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def kume(set1,set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

kume(kume1,kume2)



