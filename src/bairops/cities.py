"""The 30 Bangladeshi locations in the BAirOps dataset (id, name, latitude, longitude)."""

CITIES = [
    {"city_id": 7701354, "name": 'Azimpur', "lat": 23.7298, "lon": 90.3854},
    {"city_id": 1185283, "name": 'Badarganj', "lat": 25.67419, "lon": 89.05377},
    {"city_id": 1185274, "name": 'Baniachang', "lat": 24.51863, "lon": 91.35787},
    {"city_id": 1336137, "name": 'Barisāl', "lat": 22.70497, "lon": 90.37013},
    {"city_id": 1209562, "name": 'Bera', "lat": 24.07821, "lon": 89.63262},
    {"city_id": 1185263, "name": 'Bhairab Bāzār', "lat": 24.0524, "lon": 90.9764},
    {"city_id": 1185262, "name": 'Bherāmāra', "lat": 24.02452, "lon": 88.99234},
    {"city_id": 1336136, "name": 'Bhola', "lat": 22.68759, "lon": 90.64403},
    {"city_id": 1477498, "name": 'Bhāndāria', "lat": 22.48898, "lon": 90.06273},
    {"city_id": 1185260, "name": 'Bhātpāra Abhaynagar', "lat": 23.01472, "lon": 89.43936},
    {"city_id": 1337233, "name": 'Bogra', "lat": 24.85098, "lon": 89.37108},
    {"city_id": 1210565, "name": 'Burhānuddin', "lat": 22.49518, "lon": 90.72391},
    {"city_id": 1185276, "name": 'Bājitpur', "lat": 24.21623, "lon": 90.95002},
    {"city_id": 1185270, "name": 'Bāndarban', "lat": 22.19534, "lon": 92.21946},
    {"city_id": 1207047, "name": 'Char Bhadrāsan', "lat": 23.30916, "lon": 90.22698},
    {"city_id": 1185249, "name": 'Chhāgalnāiya', "lat": 23.02475, "lon": 91.51091},
    {"city_id": 1185254, "name": 'Chhātak', "lat": 25.03852, "lon": 91.66958},
    {"city_id": 1185247, "name": 'Chilmāri', "lat": 25.55613, "lon": 89.67097},
    {"city_id": 1205733, "name": 'Chittagong', "lat": 22.3384, "lon": 91.83168},
    {"city_id": 1185186, "name": 'Comilla', "lat": 23.46186, "lon": 91.18503},
    {"city_id": 1336134, "name": 'Cox’s Bāzār', "lat": 21.43973, "lon": 92.00955},
    {"city_id": 1185241, "name": 'Dhaka', "lat": 23.7104, "lon": 90.40744},
    {"city_id": 1203891, "name": 'Dinājpur', "lat": 25.62745, "lon": 88.63779},
    {"city_id": 1185100, "name": 'Dohār', "lat": 23.59311, "lon": 90.14251},
    {"city_id": 1203344, "name": 'Farīdpur', "lat": 23.60612, "lon": 89.84064},
    {"city_id": 6414184, "name": 'Fatikchari', "lat": 22.68768, "lon": 91.78123},
    {"city_id": 1185224, "name": 'Feni', "lat": 23.0144, "lon": 91.3966},
    {"city_id": 1185218, "name": 'Gafargaon', "lat": 24.432, "lon": 90.5585},
    {"city_id": 7921384, "name": 'Gaibandha', "lat": 25.3293, "lon": 89.5438},
    {"city_id": 1185210, "name": 'Gaurnadi', "lat": 22.97372, "lon": 90.22376},
]

CITY_BY_NAME = {c["name"].lower(): c for c in CITIES}
CITY_BY_ID = {c["city_id"]: c for c in CITIES}
