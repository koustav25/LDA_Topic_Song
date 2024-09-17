import unittest
import requests
import json

class FlaskApiTestCase(unittest.TestCase):
    BASE_URL = 'http://127.0.0.1:5000'
    
    def test_get_genres(self):
        response = requests.get(f'{self.BASE_URL}/genres')
        self.assertEqual(response.status_code, 200)
        genres = response.json()
        self.assertIsInstance(genres, list)
        print("Test /genres passed")

    def test_get_songs_by_genre(self):
        # Test with specific genres
        data = {"genres": ["Rock", "Pop"]}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'{self.BASE_URL}/songs_by_genre', data=json.dumps(data), headers=headers)
        self.assertEqual(response.status_code, 200)
        songs = response.json()
        self.assertIsInstance(songs, list)
        self.assertTrue(len(songs) <= 20)  # Ensure that the returned list is not longer than 20 items
        print("Test /songs_by_genre passed")

    def test_get_songs_by_topic(self):
        # Test with a specific topic
        data = {"topic": "Songs on hope and despair"}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'{self.BASE_URL}/songs_by_topic', data=json.dumps(data), headers=headers)
        self.assertEqual(response.status_code, 200)
        songs = response.json()
        self.assertIsInstance(songs, list)
        self.assertTrue(len(songs) <= 20)  # Ensure that the returned list is not longer than 20 items
        print("Test /songs_by_topic passed")
    
    def test_get_songs_by_genre_and_topic(self):
        # Test with a specific genre and topic
        data = {"genre": "Rock", "topic": "Songs on hope and despair"}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f'{self.BASE_URL}/songs_by_genre_and_topic', data=json.dumps(data), headers=headers)
        self.assertEqual(response.status_code, 200)
        songs = response.json()
        self.assertIsInstance(songs, list)
        self.assertTrue(len(songs) <= 20)  # Ensure that the returned list is not longer than 20 items
        print("Test /songs_by_genre_and_topic passed")



if __name__ == '__main__':
    unittest.main()
