import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.widget import Widget
import numpy as np
import cv2
import tensorflow as tf

class CanvasWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drawing = False
        self.points = []
        
        # Gambar latar belakang putih
        with self.canvas:
            Color(1, 1, 1, 1)  # Latar belakang putih
            self.rect = Rectangle(pos=self.pos, size=self.size)
        
        # Bind untuk memperbarui persegi panjang saat posisi/ukuran berubah
        self.bind(pos=self.update_rect, size=self.update_rect)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.drawing = True
            self.points = [touch.pos]
            return True
        return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.drawing and self.collide_point(*touch.pos):
            self.points.append(touch.pos)
            
            # Gambar segmen garis
            with self.canvas:
                Color(0, 0, 0, 1)  # Warna hitam
                Line(points=self.points, width=15)
            return True
        return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        if self.drawing:
            self.drawing = False
            return True
        return super().on_touch_up(touch)
    
    def clear_canvas(self):
        # Bersihkan dan gambar ulang latar belakang
        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 1)  # Latar belakang putih
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)
    
    def get_image(self):
        """Dapatkan kanvas sebagai gambar OpenCV"""
        # Ekspor kanvas ke PNG sementara
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        self.export_to_png(temp_path)
        
        # Baca gambar dengan OpenCV
        img = cv2.imread(temp_path)
        
        # Hapus file sementara
        os.unlink(temp_path)
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return gray

class DrawingWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Label hasil pengenalan
        self.result_label = Label(text='Gambar digit (0-9)', 
                                 font_size=24, 
                                 size_hint_y=None, 
                                 height=100)
        
        # Kanvas untuk menggambar
        self.canvas_widget = CanvasWidget()
        
        # Tata letak tombol
        button_layout = GridLayout(cols=2, size_hint_y=None, height=100)
        clear_btn = Button(text='Bersihkan')
        clear_btn.bind(on_press=self.clear_canvas)
        recognize_btn = Button(text='Kenali')
        recognize_btn.bind(on_press=self.recognize_digit)
        
        button_layout.add_widget(clear_btn)
        button_layout.add_widget(recognize_btn)
        
        self.add_widget(self.result_label)
        self.add_widget(self.canvas_widget)
        self.add_widget(button_layout)
        
        # Inisialisasi model pengenalan
        self.model = self.create_model()
    
    def clear_canvas(self, instance):
        self.canvas_widget.clear_canvas()
        self.result_label.text = 'Gambar digit (0-9)'
    
    def recognize_digit(self, instance):
        # Dapatkan gambar yang digambar
        img = self.canvas_widget.get_image()
        
        if img is not None and self.model is not None:
            # Pra proses dan prediksi
            processed_img = self.preprocess_image(img)
            predictions = self.model.predict(processed_img)
            prediction = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            self.result_label.text = f'Diprediksi: {prediction} (Kepercayaan: {confidence:.2f})'
        else:
            self.result_label.text = 'Model tidak tersedia'
    
    def create_model(self):
        """Load model pengenalan digit yang telah dilatih"""
        try:
            model = tf.keras.models.load_model('digit_model.h5')
            print("Model berhasil dimuat.")
            return model
        except:
            print("Model tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
            return None
    
    def preprocess_image(self, img):
        """Praproses gambar untuk pengenalan dengan model CNN"""
        # Pastikan grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # OTSU threshold untuk binerisasi optimal
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilation untuk menebalkan digit
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Temukan kontur terbesar
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Ambil kontur terbesar
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter untuk menghindari noise
            if w > 5 and h > 5 and cv2.contourArea(cnt) > 30:
                # Crop digit
                digit = thresh[max(0, y-2):min(thresh.shape[0], y+h+2), 
                              max(0, x-2):min(thresh.shape[1], x+w+2)]
                
                # Buat square dengan padding yang cukup
                h_d, w_d = digit.shape
                side = max(w_d, h_d)
                padded = np.zeros((side + 10, side + 10), dtype=np.uint8)
                offset_x = (side - w_d + 10) // 2
                offset_y = (side - h_d + 10) // 2
                padded[offset_y:offset_y+h_d, offset_x:offset_x+w_d] = digit
                
                # Resize ke 28x28 dengan interpolasi berkualitas
                resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_CUBIC)
            else:
                resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
        
        # Normalisasi ke 0-1
        normalized = resized.astype('float32') / 255.0
        input_img = normalized.reshape(1, 28, 28, 1)
        
        return input_img

class DigitRecognitionApp(App):
    def build(self):
        self.title = 'Pengenalan Digit (0-9)'
        return DrawingWidget()

if __name__ == '__main__':
    DigitRecognitionApp().run()