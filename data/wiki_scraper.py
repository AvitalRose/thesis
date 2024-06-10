import wikipediaapi
import json
import pickle as pkl
import os
import argparse


SECTIONS_TO_FILTER = ['See also', 'External links']


def find_low_level_sections(sections):
    local_list = []
    for s in sections:
        if not s.sections:
            if s.text and s.title not in SECTIONS_TO_FILTER:
                local_list.append((s.title, s.text))
        else:
            local_list.extend(find_low_level_sections(s.sections))
    return local_list


def get_categories(page):
    """
    https://dev.to/admantium/nlp-project-wikipedia-article-crawler-classification-corpus-reader-dik
    """
    if (list(page.categories.keys())) and (len(list(page.categories.keys())) > 0):
        categories = [c.replace('Category:', '').lower() for c in list(page.categories.keys())
                      if c.lower().find('articles') == -1
                      and c.lower().find('pages') == -1
                      and c.lower().find('wikipedia') == -1
                      and c.lower().find('cs1') == -1
                      and c.lower().find('webarchive') == -1
                      and c.lower().find('dmy dates') == -1
                      and c.lower().find('short description') == -1
                      and c.lower().find('commons category') == -1
                      and c.lower().find('source attribution') == -1
                      and c.lower().find('wikidata') == -1  # general
                      and c.lower().find('citation') == -1  # general
                      and c.lower().find('citation') == -1  # general
                      and c.lower().find('use american english') == -1  # general

                      ]
        return categories


def get_sub_dict(wiki_wiki, title):
    p_wiki = wiki_wiki.page(title)
    if not p_wiki.exists():
        return None, None

    categories = get_categories(page=p_wiki)

    # filter out categories which aren't visible

    sections = p_wiki.sections
    sections_with_no_sub_sections = find_low_level_sections(p_wiki.sections)
    dict_of_sections = {}
    for sec in sections_with_no_sub_sections:
        dict_of_sections[sec[0]] = sec[1]

    return categories, dict_of_sections


def get_corpus(multiple_label, file_path):
    # LIST_OF_COUNTRIES = ['Afghanistan',
    #                      'Albania',
    #                      'Algeria',
    #                      'Andorra',
    #                      'Angola',
    #                      'Antigua and Barbuda',
    #                      'Argentina',
    #                      'Armenia',
    #                      'Australia',
    #                      'Austria',
    #                      'Azerbaijan',
    #                      'The Bahamas',
    #                      'Bahrain',
    #                      'Bangladesh',
    #                      'Barbados',
    #                      'Belarus',
    #                      'Belgium',
    #                      'Belize',
    #                      'Benin',
    #                      'Bhutan',
    #                      'Bolivia',
    #                      'Bosnia and Herzegovina',
    #                      'Botswana',
    #                      'Brazil',
    #                      'Brunei',
    #                      'Bulgaria',
    #                      'Burkina Faso',
    #                      'Burundi',
    #                      'Cabo Verde',
    #                      'Cambodia',
    #                      'Cameroon',
    #                      'Canada',
    #                      'Central African Republic',
    #                      'Chad',
    #                      'Chile',
    #                      'China',
    #                      'Colombia',
    #                      'Comoros',
    #                      'Congo, Democratic Republic of the',
    #                      'Congo, Republic of the',
    #                      'Costa Rica',
    #                      'Côte d’Ivoire',
    #                      'Croatia',
    #                      'Cuba',
    #                      'Cyprus',
    #                      'Czech Republic',
    #                      'Denmark',
    #                      'Djibouti',
    #                      'Dominica',
    #                      'Dominican Republic',
    #                      'East Timor (Timor-Leste)',
    #                      'Ecuador',
    #                      'Egypt',
    #                      'El Salvador',
    #                      'Equatorial Guinea',
    #                      'Eritrea',
    #                      'Estonia',
    #                      'Eswatini',
    #                      'Ethiopia',
    #                      'Fiji',
    #                      'Finland',
    #                      'France',
    #                      'Gabon',
    #                      'The Gambia',
    #                      'Georgia',
    #                      'Germany',
    #                      'Ghana',
    #                      'Greece',
    #                      'Grenada',
    #                      'Guatemala',
    #                      'Guinea',
    #                      'Guinea-Bissau',
    #                      'Guyana',
    #                      'Haiti',
    #                      'Honduras',
    #                      'Hungary',
    #                      'Iceland',
    #                      'India',
    #                      'Indonesia',
    #                      'Iran',
    #                      'Iraq',
    #                      'Ireland',
    #                      'Israel',
    #                      'Italy',
    #                      'Jamaica',
    #                      'Japan',
    #                      'Jordan',
    #                      'Kazakhstan',
    #                      'Kenya',
    #                      'Kiribati',
    #                      'Korea, North',
    #                      'Korea, South',
    #                      'Kosovo',
    #                      'Kuwait',
    #                      'Kyrgyzstan',
    #                      'Laos',
    #                      'Latvia',
    #                      'Lebanon',
    #                      'Lesotho',
    #                      'Liberia',
    #                      'Libya',
    #                      'Liechtenstein',
    #                      'Lithuania',
    #                      'Luxembourg',
    #                      'Madagascar',
    #                      'Malawi',
    #                      'Malaysia',
    #                      'Maldives',
    #                      'Mali',
    #                      'Malta',
    #                      'Marshall Islands',
    #                      'Mauritania',
    #                      'Mauritius',
    #                      'Mexico',
    #                      'Micronesia, Federated States of',
    #                      'Moldova',
    #                      'Monaco',
    #                      'Mongolia',
    #                      'Montenegro',
    #                      'Morocco',
    #                      'Mozambique',
    #                      'Myanmar (Burma)',
    #                      'Namibia',
    #                      'Nauru',
    #                      'Nepal',
    #                      'Netherlands',
    #                      'New Zealand',
    #                      'Nicaragua',
    #                      'Niger',
    #                      'Nigeria',
    #                      'North Macedonia',
    #                      'Norway',
    #                      'Oman',
    #                      'Pakistan',
    #                      'Palau',
    #                      'Panama',
    #                      'Papua New Guinea',
    #                      'Paraguay',
    #                      'Peru',
    #                      'Philippines',
    #                      'Poland',
    #                      'Portugal',
    #                      'Qatar',
    #                      'Romania',
    #                      'Russia',
    #                      'Rwanda',
    #                      'Saint Kitts and Nevis',
    #                      'Saint Lucia',
    #                      'Saint Vincent and the Grenadines',
    #                      'Samoa',
    #                      'San Marino',
    #                      'Sao Tome and Principe',
    #                      'Saudi Arabia',
    #                      'Senegal',
    #                      'Serbia',
    #                      'Seychelles',
    #                      'Sierra Leone',
    #                      'Singapore',
    #                      'Slovakia',
    #                      'Slovenia',
    #                      'Solomon Islands',
    #                      'Somalia',
    #                      'South Africa',
    #                      'Spain',
    #                      'Sri Lanka',
    #                      'Sudan',
    #                      'Sudan, South',
    #                      'Suriname',
    #                      'Sweden',
    #                      'Switzerland',
    #                      'Syria',
    #                      'Taiwan',
    #                      'Tajikistan',
    #                      'Tanzania',
    #                      'Thailand',
    #                      'Togo',
    #                      'Tonga',
    #                      'Trinidad and Tobago',
    #                      'Tunisia',
    #                      'Turkey',
    #                      'Turkmenistan',
    #                      'Tuvalu',
    #                      'Uganda',
    #                      'Ukraine',
    #                      'United Arab Emirates',
    #                      'United Kingdom',
    #                      'United States',
    #                      'Uruguay',
    #                      'Uzbekistan',
    #                      'Vanuatu',
    #                      'Vatican City',
    #                      'Venezuela',
    #                      'Vietnam',
    #                      'Yemen',
    #                      'Zambia',
    #                      'Zimbabwe']
    with open(os.path.join("data", file_path), "r", encoding="utf-8") as f:
        list_of_pages = f.readlines()

    list_of_pages = [x.strip("\n") for x in list_of_pages]

    results_dict = {}

    # get URL
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (avitalgroupon@example.com)', 'en')
    for title in list_of_pages:

        categories, dict_of_sections = get_sub_dict(wiki_wiki, title)
        if categories and dict_of_sections:
            if multiple_label:
                results_dict[title] = {'categories': categories, 'sections': dict_of_sections}
            else:
                results_dict[title] = {'categories': [file_path.strip(".txt")], 'sections': dict_of_sections}
        else:
            print(f"No title {title}")
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+')
    arguments = parser.parse_args()
    print(f"arguments are: {arguments}")
    multi_label = True if len(arguments.file) == 1 else False
    print(f"multi label is: {multi_label}")
    countries_dict = {}
    for filename in arguments.file:
        countries_dict.update(get_corpus(multi_label, filename))
    with open(r"data\wiki.json", "w") as f:
        json.dump(countries_dict, f, indent=4)
