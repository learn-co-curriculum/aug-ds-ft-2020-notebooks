import os
import json
import requests

class RickAndMorty:


    def __init__(self):
        self.base_url = "https://rickandmortyapi.com/api/"


    @staticmethod # decorator that tells the function to not expect 'self'
    def create_final_endpoint(base_url, endpoint):
        final_endpoint = os.path.join(base_url, endpoint)
        return final_endpoint


    def find_character(self, **params):
        """
        **params:
            name: filter by the given name.
            status: filter by the given status (alive, dead or unknown).
            species: filter by the given species.
            type: filter by the given type.
            gender: filter by the given gender (female, male, genderless or unknown)
        """
        endpoint = "character"
        final_endpoint = self.create_final_endpoint(base_url=self.base_url, 
                                                    endpoint=endpoint)
        r = requests.get(final_endpoint, params=params)
        return r.json()


    def find_all_characters_by_species(self, species="human"):
        params = {"species":species}
        r = self.find_character(params=params)
        return r 


    