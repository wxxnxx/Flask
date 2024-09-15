from flask import Flask
import sys
import locale

sys.stdout.reconfigure(encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')

app = Flask(__name__)

@app.route('/home')
def home():
   return 'HI'

if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)