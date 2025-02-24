from public_invest_api import Public
import os
from dotenv import load_dotenv

load_dotenv()

public = Public()
public.login(
    username='alexgarrett2468@gmail.com',
    password=os.getenv('PUBLICPW'),
    wait_for_2fa=True # When logging in for the first time, you need to wait for the SMS code
)

# positions = public.get_positions()
# for position in positions:
#     print(position)