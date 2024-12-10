import requests

# กำหนด API Key ของคุณ
api_key = '90a48ae5e4a541c796c28ef58d3c9c75'

# กำหนด Affiliation ID สำหรับ Chulalongkorn University
affiliation_id = '60028190'

# กำหนด URL ของ Scopus API สำหรับการค้นหาข้อมูลจาก Affiliation ID
url = f'https://api.elsevier.com/content/affiliation/affiliation_id/{affiliation_id}'

# กำหนด Headers เพื่อใส่ API Key
headers = {
    'Authorization': f'Bearer {api_key}'
}

# ส่งคำขอไปยัง Scopus API
response = requests.get(url, headers=headers)
print("eiei")
print(response.json())
print("eiei")
# ตรวจสอบสถานะการตอบกลับ

