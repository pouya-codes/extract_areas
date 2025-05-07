import requests
import time
import random
from datetime import datetime

url_visa = 'https://www.ch-edoc-reservation.admin.ch/rest/public/appointment/booking/location/422/businesscase/6837836?dateFrom=2024-10-27&size=20'
url_visa = 'https://www.ch-edoc-reservation.admin.ch/rest/public/appointment/booking/location/422/businesscase/7240568?dateFrom=2025-02-05&size=20'
url_visa = 'https://www.ch-edoc-reservation.admin.ch/rest/public/appointment/booking/location/422/businesscase/7369878?dateFrom=2025-02-22&size=20'

headers = {
    'Content-Type': 'application/json',
    'Cookie': 'XSRF-TOKEN=43bafca8-79fd-408f-8bd1-7f2f1563f54b',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
    'X-XSRF-TOKEN': '43bafca8-79fd-408f-8bd1-7f2f1563f54b',
    'token': '4ae9xmEN'
}

data = {"weekdays": 31, "morningFlag": True, "afternoonFlag": True}

count = 0
import requests
TOKEN = "6813501660:AAFFN6WlupQbZc_NrLzTYdVuR-LEJNNmtgs"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
print(requests.get(url).json())


chat_id = "73859597"
chat_id = "85316256"
message = "hello from your telegram bot"
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
print(requests.get(url).json())

while True:
    current_hour = datetime.now().hour
    # 7 <= current_hour:
    if True:
        try:
            res = requests.post(url_visa, headers=headers, json=data).json()
            print(res)
            for item in res:
                message = f"found appointment on {item['timeFrom'].replace('T', ' ')}"
                url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
                # if item['timeFrom'] <= '2025-1-10':
                print(requests.get(url).json())
            print(count)
            count += 1
            time.sleep(120 + 10 * random.randrange(1, 10))
        except Exception as e:
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={e}"
