from flask import Flask, render_template, request,redirect,url_for, jsonify,session
from conexion import Conexion


app = Flask(__name__)
app.secret_key = "hello"
DataBase = Conexion().connectionDB()

