# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

### Permasalahan Bisnis
Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Cakupan Proyek
identifikasi berbagai faktor yang mempengaruhi tingginya dropout rate tersebut. Selain itu buat business dashboard untuk membantu memonitor berbagai faktor tersebut

### Persiapan

Sumber data: 
Dataset yang digunakan adalah dataset dropout yang didapatkan dari link
https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

Data selanjutnya dimasukkan ke dalam folder gdrive pribadi dengan link
https://drive.google.com/file/d/1SV1ffv0g9s9oAoMwKgVJLaKdH09BJheS/view?usp=drive_link

Setup environment Shell:

pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt

## Business Dashboard
Dashboard yang Saya buat ini dibuat berdasarkan faktor faktor yang paling berpengaruh dengan tingkat attrition
Beberapa faktor itu diantaranya Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade, Age_at_enrollment dan Scholarship_holder
https://public.tableau.com/views/DashboardFinalAssessment/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## Menjalankan Sistem Machine Learning
Prototype dijalankan dengan membuat python file yang berisi function untuk menjalankan model berdasarkan inputan user
File2 pendukung diupload di alamat github berikut
https://github.com/hakim3189/edu_institute_problem

File-file tersebut diantaranya :
README.md --> berisi penjelasan singkat mengenai latar belakang masalah dan detail aplikasi
best_rf_model.joblib --> model yang digunakan untuk pemrosesan input hasil machine learning dari file edu_institute_problem.ipynb
edu_institute_problem.ipynb --> File python yang digunakan untuk menghasilkan model berdasarkan data yang telah didapat sebelumnya
model_utils.py --> berisi file python yang menyimpan function untuk menggunakan model yang telah kita dapat untuk mendapatkan hasil prediksi
test_model.py --> file python yang akan di load di streamlite. berisi tampilan tatap muka untuk inputan serta tombol predict status untuk menjalankan prediksi dari model berdasarkan inputan

untuk selanjutnya file test_model.py akan dijalankan di aplikasi streamlite dengan link sebagai berikut
https://testmodelpy-jxc9z4oksptc3re592q234.streamlit.app/

user bisa mencoba dengan memasukkan berbagai inputan untuk mencoba hasil dari model

## Conclusion
Berdasarkan hasil analisa dari file ini didapatkan bahwa banyak faktor yang berpengaruh terhadap kemungkinan dropout dari seorang siswa. diantaranya :
- Curricular_units_1nd_sem_approved
- Curricular_units_1nd_sem_grade
- Curricular_units_2nd_sem_approved
- Curricular_units_2nd_sem_grade
- Tuition_fees_up_to_date
- Scholarship_holder
- Age_at_enrollment


### Rekomendasi Action Items
Merujuk pada beberapa faktor penting yang kita dapati berpengaruh terhadap tingkat droupout. berikut beberapa saran yang bisa kita berikan untuk mengatasi hal ini :
1. berikan perhatian lebih terhadap siswa-siswa yang mendapatkan nilai jelek terutama di semester 1 dan 2
2. memotivasi siswa untuk mengambil lebih banyak course terutama di semester 1 dan 2 untuk membuat mereka menjadi lebih merasa terlibat dan tertarik
3. tuition dan Scholarship bisa lebih digalakkan untuk memotivasi siswa untuk lebih giat belajar

