import sqlite3

import PIL
from GPSPhoto import gpsphoto
import struct
import time

def writeTofile(data, filename, gps):
    info = gpsphoto.GPSInfo((gps[2],gps[1]),alt=int(gps[3]))
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)
    print(filename)
    photo = gpsphoto.GPSPhoto(filename)
    photo.modGPSData(info, 'exif_'+filename)
    print("Stored blob data into: ", 'exif_'+filename, "\n")

def readBlobData(file_name):
    try:
        sqliteConnection = sqlite3.connect(file_name+'.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sql_fetch_blob_query = """SELECT Data.id, image, gps from Data inner join Node on Data.id = Node.id"""
        cursor.execute(sql_fetch_blob_query)
        record = cursor.fetchall()
        for row in record:
            try:
               gps = struct.unpack('dddddd',row[2])
            except:
               continue
            print("Id = ", row[0], "GPS = ", gps)
            photo = row[1]
            photoPath =  file_name+'_'+str(row[0]) + ".jpg"
            writeTofile(photo, photoPath, gps)
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to read blob data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("sqlite connection is closed")

if __name__=="__main__":
   readBlobData('save_all')
