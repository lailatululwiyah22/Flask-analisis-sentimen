from datetime import datetime
from app import db


class Laporan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    telephone = db.Column(db.String(120))
    alamat = db.Column(db.String(128))
    tujuan = db.Column(db.String(128))
    opini = db.Column(db.String(128))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    posts = db.relationship('Post', backref='author', lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)


class Tanggapan(db.Model):
    id_tanggapan = db.Column(db.Integer, primary_key=True)
    isi_balasan = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    id = db.Column(db.Integer, db.ForeignKey('laporan.id'))

    def __repr__(self):
        return '<Tanggapan {}>'.format(self.isi_balasan)