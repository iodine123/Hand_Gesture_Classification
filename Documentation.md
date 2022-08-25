# Hand Gesture Classification
## Domain Proyek
Pada percobaan kali ini kita akan melatih sebuah model dengan menggunakan library tensorflow dan python untuk membuat sebuah model yang dapat mengklasifikasi bentuk tangan. Input dari model berupa gambar. Kita akan membuat sebuah model dengan menggunakan arsitektur VGG16. Kita akan mengamati hasil akurasi dan presisi dari model setelah di training menggunakan dataset yang kita miliki.

## Dataset

Dataset merupakan data gambar yang berjumlah 2189 yang dibagi dalam 3 kelas, yaitu tangan berbentuk gunting, batu dan kertas. Dataset ini akan dibagi menjadi 2 bagian yaitu training dan validation. Training sebesat 60 persen (1328) dan sisanya (876) akan digunakan sebagai validation. Berikut contoh dataset yang digunakan :

 ![1Frbe8cdOdkciOBg](https://user-images.githubusercontent.com/57628364/186568989-e3a3ec3c-39dd-4674-b8ae-8f7344ba472d.png)
 ![0NDYNEoDui7o64gU](https://user-images.githubusercontent.com/57628364/186569159-de4c4d12-9a7e-46b3-8036-a48aebc6b0f0.png)
 ![1lEpWTJDphkm3HdC](https://user-images.githubusercontent.com/57628364/186569178-015bcdb9-d4fd-4d2c-9592-8464e14859af.png)
 
Gambar tersebut memiliki ukuran 200x200 pixel dengan background hijau. Tujuan dibuatnya gambar dengan format ini adalah untuk membuat gambar yang konsisten dan dapat dikenali dengan mudah oleh komputer. Setelah data sudah siap digunakan akan dilakukan proses data generator, yaitu preprocessing pada gambar dengan tujuan untuk memproses gambar sebelum gambar digunakan untuk training. 

![01](https://user-images.githubusercontent.com/57628364/186569797-745696d0-ffa9-4f57-bd1f-93308a849e02.JPG)

Proses yang dilakukan diatas adalah dengan mambagi semua data pixel gambar dengan 255. Sehingga skala data yang ada akan menjadi lebih kecil dengan tujuan supaya komputer dapat memproses data tersebut dengan lebih cepat. Pada proses ini bisa menggunakan library bawaan tensorflow yaitu ImageDataGenerator pada class keras. Selanjutnya setelah ditentukan preprocessing pada gambar data tersebut akan digunakan untuk melakukan proses pemanggilan dan pemrosesan pada gambar. 

![02](https://user-images.githubusercontent.com/57628364/186570229-b6f5bc26-d73f-4053-b837-07dd12642c23.JPG)

Proses diatas dilakukan menggunakan fungsi bawaan dari tensorflow yaitu ImageDataGenerator.flow_from_directory(). Artinya komputer akan menentukan klasifikasi gambar dengan melihat banyaknya folder yang ada pada direktori. Oleh sebab itu kita perlu membuat folder sesuai dengan klasifikasi yang diperlukan. Hasil dari proses ini menunjukkan keterangan :

![03](https://user-images.githubusercontent.com/57628364/186572113-cf536292-7fec-45e8-b1af-a480e68f9bc0.JPG)

## Proses Training

### Pembuatan model

Model yang digunakan pada percobaan ini adalah model dengan arsitektur VGG16 (Visual Geometry Group). Model ini mengandung beberapa layer. Contoh gambar yang saya ambil dari [link](https://www.researchgate.net/figure/Gambar-4-Arsitektur-VGG16-9_fig1_350115831) menunjukkan bagaimana arsitektur dari VGG16 ini dibuat.

![04](https://user-images.githubusercontent.com/57628364/186572701-62dddad7-ec79-4b2f-9511-0e08b7a46615.png)

Apabila dilihat pada gambar diatas menunjukkan bahwa arsitektur ini terdiri dari 2 bagian penting, yaitu convolutional layer dan fully connected layer. Pada bagian convolutional layer model akan melakukan proses konvolusi. Apabila anda tertarik untuk mengetahui tentang proses konvolusi anda bisa membaca pada [link](https://medium.com/@alifkurniawan/operasi-konvolusi-f9d0101b5bbc) berikut ini. Ada beberapa lapisan layer konvolusi. Apabila diamati setiap selesai dilakukan 2 kali proses konvolusi dilakukan proses maxpool layer. Maxpool adalah sebuah cara untuk mereduksi gambar sehingga gambar dapat diproses dengan lebih cepat. Apabila ingin mengetahui lebih dalam tentang pooling layer anda bisa membuka [link](https://medium.com/nodeflux/mengenal-convolutional-layer-dan-pooling-layer-3c6f5c393ab2) berikut. Bagian full connected layer adalah bagian yang berisi lapisan deep learning. Sehingga karena gambar adalah matriks yang berbentuk array 2 dimensi, maka kita perlu melakukan proses flatten, yaitu mengubah objek 2 dimensi menjadi bentuk 1 dimensi. Kemudian dari array 1 dimensi inilah yang digunakan untuk proses klasifikasi gambar. Intinya adalah layer konvolusi digunakan sebagai features extraction atau pengenalan fitur, atau lebih jelasnya pengenalan setiap keunikan pada gambar, seperti tepi, bentuk, warna, dll. Fully Connected digunakan untuk mengenali gambar menggunakan input fitur dari layer sebelumnya. Oke, langsung saja kita gas untuk membuat modelnya. Untuk programnya seperti dibawah:

![04](https://user-images.githubusercontent.com/57628364/186574224-41019c8b-08c3-48b2-a1c8-502f26b85180.JPG)

![05](https://user-images.githubusercontent.com/57628364/186574251-cb62960e-4f1b-4632-96f8-1e77a6a2d500.JPG)

Kita menggunakan fungsi bawaan keras untuk membuat model, yaitu models.Sequential(). Mari kita sesuaikan dengan arsitektur VGG16 yang sudah kita bahas sebelumnya, namun yang perlu diperhatikan adalah layer terakhir yang merupakan layer klasifikasi. Layer ini akan berjumlah sama dengan banyaknya klasifikasi yang kita gunakan. Abaikan tahap dropout, apabila anda ingin mengetahui apa itu dropout anda bisa mempelajarinya pada [tautan](https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e) berikut ini. Kemudian model akan kita compile menggunakan parameter yang saya gunakan. Untuk lebih jelasnya anda bisa pelajari mengenai hyperparameter yang saya gunakan pada model tersebut, karena saya rasa akan terlalu panjang apabila saya menjelaskan hal tersebut hehe. Sedikit informasi pada [link](https://codingstudio.id/hyperparameter-tuning/) berikut sebagaii tambahan pemahaman mengenai hyperparameter. Oke, setelah berhasil membuat model akan kita lanjut pada proses training menggunakan dataset yang kita siapkan sebelumnya.

### Proses Training
Pada proses training saya membuat sebuah callbacks, namun anda dapat mengabaikan callbacks ini. Namun, apabila anda ingin mengetahui lebih jelas tentang proses ini anda bisa membaca pada [tautan](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) berikut ini. Proses training dilakukan menggunakan fungsi fit(). Lebih jelasnya lihat pada gambar dibawah :

![06](https://user-images.githubusercontent.com/57628364/186575818-c052893d-995b-44cb-963e-f116b3a69d39.JPG)


Baik, saya akan mencoba menjelaskan dengan lebih detail tentang parameter yang digunakan pada fungsi fit() tersebut. \
1. train_generator, adalah sebuah variabel yang sudah berisi informasi dari ImageDataGenerator yang kita buat sebelumnya.
2. step_per_epoch, adalah step yang kita tentukan pada setiap epoch, jadi dalam satu epoch akan terdapat 32 step.
3. epochs, adalah banyaknya epoch, atau proses pembelajaran yang dilakukan pada semua data yang ada.
4. validation_data, sama seperti train_generator, namun data ini adalah data yang digunakan sebagai validasi model. Model akan diuji dengan data yang ada pada variabel ini.
5. validation_step, sama seperti step_per_epoch
6. verbose, adalah informasi yang ditampilkan pada saat proses training. Kita bisa men set nya kedalam 0,1, atau 2. Semakin besar nilainya maka akan menampilkan semakin banyak informasi.
7. callbacks, merupakan callback yang digunakan pada proses training

Hasil dari training bisa dilihat pada gambar dibawah:

![07](https://user-images.githubusercontent.com/57628364/186577030-70457ffa-2b91-45ba-9204-6c96b1299107.JPG)

Bisa dilihat gambar diatas adalah 5 epoch terakhir pada proses training. Bisa dilihat bahwa model memiliki akurasi sebesar 98 persen. Agar kita bisa melihat dan memahami data tersebut dengan baik saya akan mencoba plot hasil akurasi tiap epoch nya pada gambar dibawah :

![08](https://user-images.githubusercontent.com/57628364/186577598-6f5d99fe-ae61-43f8-a27a-47c5f7f56b4a.JPG)

Dari grafik diatas kita bisa melihat perkembangan akurasi dan loss model pada setiap epoch nya. 

## Pengujian Pada Gambar

Gambar dibawah menunjukkan hasil pengujian pada gambar :

![12](https://user-images.githubusercontent.com/57628364/186578760-91491ee8-309c-4cf2-96eb-97c5c6863d3a.JPG)

![13](https://user-images.githubusercontent.com/57628364/186578776-1d598366-a00e-4fad-ba74-dca39edd284a.JPG)

![14](https://user-images.githubusercontent.com/57628364/186578799-52ac2d27-c81e-4b43-979a-12e8ae8de84a.JPG)


Akhirnya model yang sudah kita buat dan kita training menggunakan dataset yang kita miliki sudah berhasil digunakan untuk mengklasifikasi jenis bentuk tangan. Apabila ada pertanyaan lebih lanjut yang bisa japri saya di instagram : @iodinehanifan.

