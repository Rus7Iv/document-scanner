from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image
import base64
from Helpers import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('Нет части файла', 'warning')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('Изображение для загрузки не выбрано', 'warning')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		
		filestr = request.files['file'].read()
		npimg = np.frombuffer(filestr, np.uint8)
		image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
		ratio = image.shape[0] / 500.0
		orig = image.copy()
		image = Helpers.resize(image, height = 500)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		edged = cv2.Canny(gray, 75, 200)

		cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = Helpers.grab_contours(cnts)
		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:
				screenCnt = approx
				break

		if 'screenCnt' in locals():
			cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
			warped = Helpers.transform(orig, screenCnt.reshape(4, 2) * ratio)
			img = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
			file_object = io.BytesIO()
			img = Image.fromarray(Helpers.resize(img, width=500))
			img.save(file_object, 'PNG')
			base64img = "data:image/png;base64," + base64.b64encode(file_object.getvalue()).decode('ascii')
			flash('Документ успешно отсканирован', 'success')
			return render_template('upload.html', image=base64img )
		else:
			flash('Ошибка сканирования', 'danger')
			return redirect(request.url)

	else:
		flash('Разрешены типы изображений: png, jpg, jpeg', 'warning')
		return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)