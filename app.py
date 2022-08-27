import io
from unicodedata import name
from unittest import result
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, send_file
from datetime import datetime
from fileinput import filename
import os
from matplotlib import container
import pandas
from sentiment import lower, remove_punctuation, remove_stopwords, stem_text, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.naive_bayes import MultinomialNB
import pymysql.cursors
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import StringIO
import csv


app = Flask(__name__, template_folder='template')
app.static_folder='static'
app.secret_key='aww'


conn = cursor = None
#fungsi koneksi database
def openDb():
   global conn, cursor
   import pymysql
   
   conn = pymysql.connect(host='localhost',user='root',password='',database='db_lapor',charset='utf8mb4')
   cursor = conn.cursor()
  
#fungsi untuk menutup koneksi
def closeDb():
   global conn, cursor
   #cursor.close()
   conn.close()


#VISUALISASI
@app.route('/grafik-visual', methods=['GET', 'POST'])
def visual_grafik():
    openDb()
    cursor = conn.cursor()
    query = "SELECT ket_sentimen, count(ket_sentimen) from keterangan_sentimen GROUP BY(ket_sentimen)"
    cursor.execute(query) 
    data = cursor.fetchall()
    #query="SELECT class,COUNT( * ) number FROM student GROUP BY class"    
    df = pandas.read_sql(query,cursor)
    lb= [row for row in df['ket_sentimen']] # Labels of graph
    plot=df.plot.pie(title="Klasfikasi sentimen ",y='number',labels=lb,autopct='%1.0f%%')
    closeDb()
    return render_template('dashboard-user.html', data=data, plot=plot, lb=lb)

#LANDING PAGE
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')

@app.route('/layout', methods=['GET', 'POST'])
def layout_admin():
    return render_template('layout_admin.html')

@app.route('/tentang', methods=['GET', 'POST'])
def about():
    return render_template('tentang.html')





@app.route('/dashboard-user', methods=['GET', 'POST'])
def dashboard_user():
    openDb()

    cursor = conn.cursor()
    number_of_rows = "SELECT ket_sentimen, count(ket_sentimen) from keterangan_sentimen GROUP BY(ket_sentimen)"
    cursor.execute(number_of_rows) 
    data = cursor.fetchall()

    number_of_rows2 = "SELECT ket_topik, count(ket_topik) from keterangan_sentimen GROUP BY(ket_topik)"
    cursor.execute(number_of_rows2) 
    data2 = cursor.fetchall()
  
    closeDb()
    return render_template("dashboard-user.html", data=data, data2=data2)

@app.route('/visualisasi', methods=['GET', 'POST'])
def visual():
    data = None
    openDb()
    number_of_rows = cursor.execute("SELECT hasilsvm from riwayat_testing")
    closeDb()
    return render_template("echarts.html", number_of_rows=number_of_rows)

@app.route('/homeadmin', methods=['GET', 'POST'])
def home_admin(): 
    openDb()
    if session.get('email'):  
        return redirect(url_for('homeadmin'))
    text = pandas.read_csv('upload/kelar banget.csv', encoding='latin-1')
    text.dropna(axis=0)
    positif, negatif= text['label'].value_counts()
    lainnya, pemerintahan, kesehatan, wisata= text['topik'].value_counts()
    jumlah = lainnya + pemerintahan + kesehatan + wisata

    #klasifikasi data
    cursor = conn.cursor()
    sentimen = "SELECT ket_sentimen, count(ket_sentimen) from keterangan_sentimen GROUP BY(ket_sentimen)"
    cursor.execute(sentimen) 
    datas = cursor.fetchall()
    topik = "SELECT ket_topik, count(ket_topik) from keterangan_sentimen GROUP BY(ket_topik)"
    cursor.execute(topik) 
    datas2 = cursor.fetchall()

    #berita
    berita = cursor.execute("SELECT * FROM beritas")

    #pengaduan
    pengaduan = cursor.execute("SELECT * FROM pengaduans")
    closeDb()
    return render_template('home_admin.html', pengaduan=pengaduan ,berita=berita ,datas=datas, datas2=datas2,positif=positif, jumlah=jumlah, negatif=negatif, pemerintahan=pemerintahan, kesehatan=kesehatan, wisata=wisata, lainnya=lainnya)

@app.route('/adminhome', methods=['GET', 'POST'])
def homeadmin():
    openDb()
    text = pandas.read_csv('upload/kelar banget.csv', encoding='latin-1')
    text.dropna(axis=0)
    positif, negatif= text['label'].value_counts()
    lainnya, pemerintahan, kesehatan, wisata= text['topik'].value_counts()
    jumlah = lainnya + pemerintahan + kesehatan + wisata

    #klasifikasi data
    cursor = conn.cursor()
    sentimen = "SELECT ket_sentimen, count(ket_sentimen) from keterangan_sentimen GROUP BY(ket_sentimen)"
    cursor.execute(sentimen) 
    datas = cursor.fetchall()
    topik = "SELECT ket_topik, count(ket_topik) from keterangan_sentimen GROUP BY(ket_topik)"
    cursor.execute(topik) 
    datas2 = cursor.fetchall()

    #berita
    berita = cursor.execute("SELECT * FROM beritas")

    #pengaduan
    pengaduan = cursor.execute("SELECT * FROM pengaduans")
    closeDb()
    return render_template('home_admin.html', positif=positif, jumlah=jumlah, negatif=negatif, pemerintahan=pemerintahan, kesehatan=kesehatan, wisata=wisata, lainnya=lainnya, datas=datas, datas2=datas2, berita=berita, pengaduan=pengaduan)


@app.route('/homeuser', methods=['GET', 'POST'])
def home_user():

    if session.get('email'):  
        return redirect(url_for('homeUser'))
   
    return render_template('homeuser.html')

@app.route('/home-user', methods=['GET', 'POST'])
def homeUser():
    return render_template('homeuser.html')

#@app.route('/lapor', methods=['GET', 'POST'])
#def formLapor():
    #error={}
    #if request.method == 'POST':
        #nama        = request.form['nama']
        #print("Name =", nama)
        #print("Phone =", request.form["telephone"])
        #print("Tujuan =", request.form["tujuan"]) 
        #print("Opini =", request.form["opini"])   
        #if not nama[0].isalpha():
                #error['nama']= ['nama tidak boleh angka']
    #return render_template('form.html', error=error)





#UPLOAD DATA SENTIMENT DI HOME ADMIN
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER']='upload'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/training_admin', methods=['GET', 'POST'])
def uploaddata():
    if request.method == 'GET':
        return render_template('training_admin.html')
    
    elif request.method == 'POST':
        file = request.files['file']
        
        if 'file' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)
      

        if file and allowed_file(file.filename):
            file.filename = "bisa ya tuhan.csv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            text = pandas.read_csv('upload/bisa ya tuhan.csv', encoding='latin-1')
         
            return render_template('training_admin.html',tables=[text.to_html()])

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    return render_template ('preposs_admin.html')

@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    text = pandas.read_csv('upload/bisa ya tuhan.csv', encoding='latin-1')
    text.dropna(axis=0)
    text.drop(['text','label'], axis=1, inplace=True)

    
    text['text'] = text['text'].map(lambda x: lower(x))
    text['text'] = text['text'].map(lambda x: remove_punctuation(x))
    text['text'] = text['text'].map(lambda x: remove_stopwords(x))
    text['text'] = text['text'].map(lambda x: stem_text(x))

    text.to_csv('upload/fix udah clear.csv', index = False, header = True)

    return render_template('prepross_admin.html',tables=[text.to_html()])

#VEKTORISASI DI HOME ADMIN
@app.route('/tfidfpage', methods=['GET', 'POST'])
def tfidfpage():
    text = pandas.read_csv('upload/bisa ya yuhan.csv', encoding='latin-1')
    text.dropna(axis=0)
    positif, negatif= text['label'].value_counts()
    total = positif + negatif
    
    return render_template ('tfidf_admin.html', total=total, positif=positif, negatif=negatif)
def data(text):
    text['label'] = text['label'].map({'positif': 1, 'negatif': 0})
    X = text['text']
    y = text['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)
    return X_train, X_test, y_train, y_test

@app.route('/tfidf', methods=['GET', 'POST'])
def tfidf():
    text = pandas.read_csv('upload/bisa ya tuhan.csv', encoding='latin-1')
    text.dropna(axis=0)
    positif, negatif= text['label'].value_counts()
    total = positif + negatif

    X_train, X_test, y_train, y_test= data(text)
    global str

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    bow_vectorizer = CountVectorizer()
    X_train = bow_vectorizer.fit_transform(text['text'])
  
    #Saving vectorizer to disk
    pickle.dump(vectorizer, open('upload/vectorizerfix.model','wb'))
    pickle.dump(bow_vectorizer, open('upload/bowfix.model','wb'))
    
    #pickle.dump(vectorizer, open('upload/vectorizer.model','wb'))
    #pickle.dump(bow_vectorizer, open('upload/bow.model','wb'))

    return render_template ('tfidf_admin.html', X_train=X_train, X_test=X_test, total=total, positif=positif, negatif=negatif)

#KLASIFIKASI DI HOME ADMIN
@app.route('/klasifikasinb_admin', methods=['GET', 'POST'])
def klasifikasinb_admin():

    return render_template ('klasifikasinb_admin.html')


@app.route('/klasifikasinb', methods=['GET', 'POST'])
def klasifikasinb():
    
    # Loading model to compare the results
    #vectorizer = pickle.load(open('upload/vectorizer.model','rb'))

    vectorizer = pickle.load(open('upload/vectorizerfix.model','rb'))

    text = pandas.read_csv('upload/bisa ya tuhan.csv', encoding='latin-1')

    X_train, X_test, y_train, y_test = data(text)

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Process of making models Klasifikasi SVM LINEAR
    model = MultinomialNB()
    model.fit(X_train,y_train)
    model_predict = model.predict(X_test)

    #Saving vectorizer to disk
    #pickle.dump(model, open('upload/model.model','wb'))

    pickle.dump(model, open('upload/modelfix.model','wb'))
    from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
    # f1_score
    f1_score_linear = f1_score(y_test, model_predict)
    

    # accuracy score
    accuracy_score_linear = accuracy_score(y_test, model_predict)
   

    # precision score
    precision_score_linear = precision_score(y_test, model_predict)
   

    # recall score
    recall_score_linear = recall_score(y_test, model_predict)
    

    # confusion matrix
    tn_linear, fp_linear, fn_linear, tp_linear = confusion_matrix(y_test, model_predict).ravel()
   

    return render_template ('klasifikasinb_admin.html', f1_score_linear=f1_score_linear, accuracy_score_linear=accuracy_score_linear, precision_score_linear=precision_score_linear, recall_score_linear=recall_score_linear, 
    tn_linear=tn_linear, fp_linear=fp_linear, fn_linear=fn_linear, tp_linear=tp_linear)

@app.route('/form', methods=['GET', 'POST'])
def form():

    return render_template ('tesmodel.html')

@app.route('/tesdb', methods=['GET', 'POST'])
def testingDB():

    return render_template ('tesmodel-db.html')

@app.route('/tesmodel', methods=['GET', 'POST'])
def tesmodel():
    # Loading model to compare the results
    #model_predict = pickle.load(open('upload/model.model','rb'))
    #vectorizer = pickle.load(open('upload/vectorizer.model','rb'))

    model_predict = pickle.load(open('upload/modelfix.model','rb'))
    vectorizer = pickle.load(open('upload/vectorizerfix.model','rb'))
    model_predict_topic = pickle.load(open('upload/modeltopic.model','rb'))
    vectorizer_topic = pickle.load(open('upload/vectorizertopic.model','rb'))

    text = request.form['text']
    original_text = request.form['text']

    #KLASIFIKASI SENTIMEN
    hasilprepro = preprocess_data(text)
    hasiltfidf = vectorizer.transform([hasilprepro])
    # cek prediksi dari kalimat
    hasilsvm = model_predict.predict(hasiltfidf)

    #KLASIFIKASI TOPIK
    hasil2 = vectorizer_topic .transform([hasilprepro])
    # cek prediksi dari kalimat
    final = model_predict_topic.predict(hasil2)
    
    return render_template ('tesmodel.html', original_text=original_text, hasilprepro=hasilprepro, hasilsvm=hasilsvm, hasil2=hasil2, final=final)

#fungsi view index() untuk menampilkan data dari database
@app.route('/tabeluji', methods=['GET', 'POST'])
def index_uji():   
   openDb()
   container = []
   sql = "SELECT * FROM keterangan_sentimen"
   cursor.execute(sql)
   results = cursor.fetchall()
   for data in results:
      container.append(data)
   closeDb()
   return render_template('uji-db.html', container=container)


@app.route('/tesuji', methods=['GET','POST'])
def tes_uji():
    
     
    model_predict = pickle.load(open('upload/model.json','rb'))
    vectorizer = pickle.load(open('upload/vectorizer.json','rb')) 
    #model topik
    model_predict_topic = pickle.load(open('upload/modeltopic.model','rb'))
    vectorizer_topic = pickle.load(open('upload/vectorizertopic.model','rb'))

    if request.method == 'POST':
        opini = request.form['opini']
        klasifikasi_manual = request.form['klasifikasi_manual']
        prediksi_topik = request.form['prediksi_topik']
      
        hasilprepro = preprocess_data(opini)
        hasiltfidf = vectorizer.transform([hasilprepro])
        # cek prediksi dari kalimat
        klasifikasi_sistem = model_predict.predict(hasiltfidf)

        hasilvect = vectorizer_topic.transform([hasilprepro])
        # cek prediksi dari kalimat
        klasifikasi_topik = model_predict_topic.predict(hasilvect)
       
        
        openDb()
        sql = "INSERT INTO ujis (opini, klasifikasi_manual,prediksi_topik,hasilprepro, klasifikasi_sistem, klasifikasi_topik) VALUES (%s, %s, %s, %s,%s, %s)"
        val = (opini, klasifikasi_manual,prediksi_topik, hasilprepro, klasifikasi_sistem, klasifikasi_topik)
        cursor.execute(sql, val)
        conn.commit()
        closeDb()
        return redirect(url_for('index_uji'))
    else:
        return render_template('tesmodel-db.html', klasifikasi_sistem=klasifikasi_sistem)



#KLASIFIKASI TOPIK
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/topik_admin', methods=['GET', 'POST'])
def uploaddata2():
    if request.method == 'GET':
        return render_template('topik_admin.html')
    
    elif request.method == 'POST':
        file = request.files['file']
        
        if 'file' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)
      

        if file and allowed_file(file.filename):
            file.filename = "kelar banget.csv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            text = pandas.read_csv('upload/kelar banget.csv', encoding='latin-1')
         
            return render_template('topik_admin.html',tables=[text.to_html()])
            
@app.route('/tfidftopic', methods=['GET', 'POST'])
def tfidftopic():
    text = pandas.read_csv('upload/kelar banget.csv', encoding='latin-1')
    text.dropna(axis=0)
    positif, negatif= text['label'].value_counts()
    lainnya, pemerintahan, kesehatan, wisata= text['topik'].value_counts()
    total = positif + negatif
    jumlah = lainnya + pemerintahan + kesehatan + wisata
    
    return render_template ('tfidftopic_admin.html', total=total, positif=positif, negatif=negatif, jumlah=jumlah, lainnya=lainnya, pemerintahan=pemerintahan, kesehatan=kesehatan, wisata=wisata)

def data(text):
    text['topik'] = text['topik'].map({'lainnya': 1, 'pemerintahan': 2, 'kesehatan':3, 'wisata':4})
    X = text['text']
    y = text['topik']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=142)
    return X_train, X_test, y_train, y_test


@app.route('/tfidftopictest', methods=['GET', 'POST'])
def tfidftopictest():
    text = pandas.read_csv('upload/kelar banget.csv', encoding='latin-1')
    text.dropna(axis=0)
    positif, negatif= text['label'].value_counts()
    lainnya, pemerintahan, kesehatan, wisata= text['topik'].value_counts()
    total = positif + negatif
    jumlah = lainnya + pemerintahan + kesehatan + wisata

    X_train, X_test, y_train, y_test= data(text)
    global str

    vectorizer_topic = TfidfVectorizer(use_idf=False, lowercase=True)
    X_train = vectorizer_topic.fit_transform(X_train)
    X_test = vectorizer_topic.transform(X_test)

    bow_vectorizer_topik = CountVectorizer()
    X_train = bow_vectorizer_topik.fit_transform(text['text'])
  
    #Saving vectorizer to disk
    pickle.dump(vectorizer_topic, open('upload/vectorizertopicfix.model','wb'))
    pickle.dump(bow_vectorizer_topik, open('upload/bowtopikfix.model','wb'))

    return render_template ('tfidftopic_admin.html', X_train=X_train, X_test=X_test, total=total, positif=positif, negatif=negatif, jumlah=jumlah, wisata=wisata, pemerintahan=pemerintahan, lainnya=lainnya, kesehatan=kesehatan)


@app.route('/klasifikasitopik_admin', methods=['GET', 'POST'])
def klasifikasitopik_admin():

    return render_template ('klasifikasitopik_admin.html')


@app.route('/klasifikasitopic', methods=['GET', 'POST'])
def klasifikasitopic():
    
    # Loading model to compare the results
    vectorizer_topic = pickle.load(open('upload/vectorizertopicfix.model','rb'))

    text = pandas.read_csv('upload/kelar banget.csv', encoding='latin-1')

    X_train, X_test, y_train, y_test = data(text)

    vectorizer_topic = TfidfVectorizer()

    X_train = vectorizer_topic.fit_transform(X_train)
    X_test = vectorizer_topic.transform(X_test)

    # Process of making models Klasifikasi SVM LINEAR
    model_topic = MultinomialNB()
    model_topic.fit(X_train,y_train)
    model_predict_topic = model_topic.predict(X_test)

    #Saving vectorizer to disk
    pickle.dump(model_topic, open('upload/modeltopicfix.model','wb'))
    from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score
    # f1_score
    f1_score_linear = f1_score(y_test, model_predict_topic, average='macro')
    

    # accuracy score
    accuracy_score_linear = accuracy_score(y_test, model_predict_topic)
   

    # precision score
    precision_score_linear = precision_score(y_test, model_predict_topic, average='macro')
   

    # recall score
    recall_score_linear = recall_score(y_test, model_predict_topic, average='macro')
    

    # confusion matrix
    
   

    return render_template ('klasifikasitopik_admin.html', f1_score_linear=f1_score_linear, accuracy_score_linear=accuracy_score_linear, precision_score_linear=precision_score_linear, recall_score_linear=recall_score_linear, 
    )


@app.route('/formtopik', methods=['GET', 'POST'])
def formtopik():

    return render_template ('testtopik.html')


@app.route('/testtopic', methods=['GET', 'POST'])
def testopic():
    # Loading model to compare the results
    model_predict_topic = pickle.load(open('upload/modeltopicfix.model','rb'))
    vectorizer_topic = pickle.load(open('upload/vectorizertopicfix.model','rb'))

    text = request.form['text']
    original_text = request.form['text']

    hasil = preprocess_data(text)
    hasil2 = vectorizer_topic .transform([hasil])
    # cek prediksi dari kalimat
    final = model_predict_topic.predict(hasil2)
    
    return render_template ('testtopik.html', original_text=original_text, hasil=hasil, final=final)





#CRUD BERITA

            
  

konten = [
    {
        'penulis': 'Raditya Dika',
        'judul': 'Postingan Pertama',
        'sinopsis': 'Ini adalah postingan pertama',
        'isi': 'Ini adalah isi dari postingan pertama',
        'tanggal': '12 Desember 2020',
        'jam': '16.00'
    }
    
]

@app.route('/artikel', methods=['GET','POST'])
def artikel():
    openDb()
    container = []
    sql = "SELECT * FROM beritas"
    cursor.execute(sql)
    results = cursor.fetchall()
    for data in results:
        container.append(data)
    
    closeDb()
    return render_template('artikel.html', container=container)
    #return render_template('artikel.html', konten=konten, judul='page')

#NEWS
@app.route('/admin_berita', methods=['GET','POST'])
def beritaAdmin():   
   openDb()
   container = []
   sql = "SELECT * FROM beritas"
   cursor.execute(sql)
   results = cursor.fetchall()
   for data in results:
      container.append(data)
   closeDb()
   return render_template('admin-berita.html', container=container,)

#fungsi view tambah() untuk membuat form tambah
@app.route('/add_berita', methods=['GET','POST'])
def tambahBerita():
   if request.method == 'POST':
      penulis = request.form['penulis']
      judul = request.form['judul']
      sinopsis = request.form['sinopsis']
      isi = request.form['penulis']
      
      openDb()
      sql = "INSERT INTO beritas (penulis, judul,sinopsis, isi) VALUES (%s, %s, %s,%s)"
      val = (penulis, judul, sinopsis, isi)
      cursor.execute(sql, val)
      conn.commit()
      closeDb()
      return redirect(url_for('beritaAdmin'))
   else:
      return render_template('tambah-berita.html')

#fungsi view edit() untuk form edit
@app.route('/edit-berita/<id>', methods=['GET','POST'])
def edit_berita(id):
   openDb()
   cursor.execute('SELECT * FROM beritas WHERE id=%s', (id))
   data = cursor.fetchone()
   if request.method == 'POST':
      
      penulis = request.form['penulis']
      judul = request.form['judul']
      sinopsis = request.form['sinopsis']
      isi = request.form['isi']
      sql = "UPDATE barang SET penulis=%s, judul=%s, sinopsis=%s isi=%s ,WHERE id=%s"
      val = (penulis, judul, sinopsis, isi)
      cursor.execute(sql, val)
      conn.commit()
      closeDb()
      return redirect(url_for('beritaAdmin'))
   else:
      closeDb()
      return render_template('.html', data=data)
#fungsi untuk menghapus data
@app.route('/hapus-berita/<id>', methods=['GET','POST'])
def hapus_berita(id):
   openDb()
   cursor.execute('DELETE FROM beritas WHERE id=%s', (id,))
   conn.commit()
   closeDb()
   return redirect(url_for('beritaAdmin'))


#LOGIN
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    
    if request.method == 'POST':
        email = request. form['email']
        password = request. form ['password']
        if email == 'admin@gmail.com' and password == 'admin':
            session ['email'] = email
            return redirect(url_for('home_admin'))
        else:
            return redirect(url_for('signin'))
    
    return render_template('signin.html')

#LOGIN
@app.route('/login-user', methods=['GET', 'POST'])
def login_user():
    
    if request.method == 'POST':
        email = request. form['email']
        password = request. form ['password']
        if email == 'latul@gmail.com' and password == 'latul':
            session ['email'] = email
            return redirect(url_for('home_user'))
        else:
            return redirect(url_for('login_user'))
    
    return render_template('login_user.html')

# LOGOUT
@app.route('/logout')
def logout():
    session.pop('user',None)
    return redirect('/signin')

@app.route('/logout-user')
def logout_user():
    session.pop('user',None)
    return redirect('/login-user')

@app.route('/form_user', methods=['GET', 'POST'])
def formUser():
    return render_template('form-user.html')

@app.route('/add-test')
def add_test():
    
    return render_template('add-testing.html')



@app.route('/cek-sentimen/<id>', methods=['GET','POST'])
def edit(id):

    model_predict = pickle.load(open('upload/modelfix.model','rb'))
    vectorizer = pickle.load(open('upload/vectorizerfix.model','rb')) 
    #model topik
    model_predict_topic = pickle.load(open('upload/modeltopicfix.model','rb'))
    vectorizer_topic = pickle.load(open('upload/vectorizertopicfix.model','rb'))
    openDb()
    cursor.execute('SELECT * FROM pengaduans WHERE id=%s', (id))
    data = cursor.fetchone()
    if request.method == 'POST':
      
        id = request.form['id']
        nama = request.form['nama']
        telephone = request.form['telephone']
        alamat = request.form['alamat']
        opini = request.form['opini']

        hasilprepro = preprocess_data(opini)
        hasiltfidf = vectorizer.transform([hasilprepro])
        # cek prediksi dari kalimat
        hasilsvm = model_predict.predict(hasiltfidf)

        hasilvect = vectorizer_topic.transform([hasilprepro])
        # cek prediksi dari kalimat
        hasil_topik = model_predict_topic.predict(hasilvect)

        sql = "INSERT INTO riwayat_testing (id,nama, telephone,alamat,opini, hasilprepro, hasilsvm,hasil_topik) VALUES (%s, %s, %s, %s,%s, %s, %s, %s)"
        val = (id,nama, telephone, alamat, opini,hasilprepro,hasilsvm,hasil_topik)
        cursor.execute(sql, val)
        conn.commit()
        closeDb()
        return redirect(url_for('data_uji'))
    else:
        closeDb()
        return render_template('cek-senti.html', data=data)

#fungsi view index() untuk menampilkan data dari database
@app.route('/data-db')
def data_uji():   
   openDb()
   container = []
   sql = "SELECT * FROM riwayat_testing"
   cursor.execute(sql)
   results = cursor.fetchall()
   for data in results: 
      container.append(data)
   closeDb()
   return render_template('data-testing.html', container=container)   

#fungsi view tambah() untuk membuat form tambah
@app.route('/add-testing-admin', methods=['GET','POST'])
def add_testing_admin():
    openDb()
    cursor.execute("SELECT * FROM riwayat_testing")
    data = cursor.fetchall()
    if request.method == 'POST':
        id_testing = request.form['id_testing']
        ket_sentimen = request.form['ket_sentimen']
        ket_topik = request.form['ket_topik']
        sql = "INSERT INTO keterangan_sentimen (id_testing,ket_sentimen ,ket_topik) VALUES (%s, %s, %s)"
        val = (id_testing,ket_sentimen,ket_topik)
        cursor.execute(sql, val)
        conn.commit()
        closeDb()
        return redirect(url_for('index_uji'))
    else:
        return render_template('add-testing.html', data=data)



#fungsi view tambah() untuk membuat form tambah
@app.route('/tambah-lapor', methods=['GET','POST'])
def tambah_lapor_user():
    if request.method == 'POST':
        nama = request.form['nama']
        telephone = request.form['telephone']
        alamat = request.form['alamat']
      
        opini = request.form['opini']

        openDb()
        
        sql = "INSERT INTO pengaduans (nama,telephone ,alamat, opini) VALUES (%s, %s, %s, %s)"
        val = (nama,telephone,alamat, opini)
        cursor.execute(sql, val)
        conn.commit()
        closeDb()
        flash("Opini anda terkirim , Terimakasih sudah mengirim pelaporan anda !")
        return redirect(url_for('formUser'))
    else:
        return render_template('form-user.html')


#CRUD PENGDUAN ADMIN
@app.route('/pengaduan-admin', methods=['GET','POST'])
def pengaduanAdmin():
    openDb()
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    container = []
    sql = "SELECT * FROM pengaduans"
    cursor.execute(sql)
    results = cursor.fetchall()
    for data in results:
        container.append(data)
    closeDb()
    return render_template('laporan_admin.html', container=container, formatted_now=formatted_now)

#CRUD PENGDUAN ADMIN
@app.route('/riwayat-pengaduan', methods=['GET','POST'])
def pengaduan_user():
    openDb()
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    container = []
    sql = "SELECT * FROM pengaduans"
    cursor.execute(sql)
    results = cursor.fetchall()
    for data in results:
        container.append(data)
    closeDb()
    return render_template('riwayat-pengaduan-user.html', container=container, formatted_now=formatted_now)



   

#fungsi untuk menghapus data
@app.route('/hapus/<id>', methods=['GET','POST'])
def hapus(id):
   openDb()
   cursor.execute('DELETE FROM pengaduans WHERE id=%s', (id,))
   conn.commit()
   closeDb()
   return redirect(url_for('pengaduanAdmin'))

#fungsi untuk menghapus data
@app.route('/hapus_uji/<id_testing>', methods=['GET','POST'])
def hapus_uji(id_testing):
   openDb()
   cursor.execute('DELETE FROM riwayat_testing WHERE id_testing=%s', (id_testing))
   conn.commit()
   closeDb()
   return redirect(url_for('data_uji'))




#DOWNLOAD
@app.route('/download/report/csv')
def download_report():
    openDb()
    cursor.execute("SELECT id, nama, telephone, alamat, opini FROM pengaduans")
    result = cursor.fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    line = ['id,nama,telephone,alamat,opini']
    writer.writerow(line)
    for row in result:
        line = [str(row['id']) + ',' + row['nama'] + ',' + row['telpehone'] + ',' + row['alamat']+ ',' + row['opini']]
        writer.writerow(line)
        output.seek(0)
        return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=pengaduan-report.csv"})
    closeDb()


if __name__ == '__main__':
    app.run(debug=True)