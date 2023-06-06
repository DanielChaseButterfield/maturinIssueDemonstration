import numpy as np
import copy
import sys
import os

cur_file_path = os.path.dirname(os.path.abspath(__file__))
root_path = '/'.join(cur_file_path.split('/')[:-2])
sys.path.append(root_path)

class AcTagParams:
    '''
    Stores the parameters of an AcTag family. It also finds the the tag family file and parses the
    data into memory.

    :param tag_family: The name of a AcTag family found in the actag_families/generated_families folder, or the path to a valid AcTag family file. See the `actag_families <https://bitbucket.org/frostlab/actag/src/master/actag_families/>`_ module for more information.
    :type tag_family: str
    :param tag_size: The length of the inner white square of the physical AcTag used in meters.
    :type tag_size: float
    '''
    def __init__(self, tag_family: str, tag_size: float) -> None:
        # See if the user passed us a path for the tag family
        if tag_family.endswith('.java'):
            self.path_to_tag_family = tag_family
            while tag_family.find("/") != -1:
                tag_family = tag_family[tag_family.find("/")+1:]
            while tag_family.find("\\") != -1:
                tag_family = tag_family[tag_family.find("\\")+1:]
            self.tag_family = tag_family[:tag_family.find(".java")]
        else:
            self.tag_family = tag_family
            path_to_actag_repo = __file__[:__file__.find("actag_detection")]
            path_to_actag_families = path_to_actag_repo + str("actag_families/generated_families/")
            self.path_to_tag_family = path_to_actag_families + self.tag_family + ".java"

        # Store inputs
        self.tag_size = tag_size

        # Verify that the tag family is in the correct input format
        if not self.tag_family.startswith('AcTag') or self.tag_family.find("h") == -1:
            raise ValueError("AcTag family must be in the format " + 
                             "'AcTag<#data_bits>h<#hamming_dist>', eg. AcTag24h8")
        
        # Calculate other useful parameters based on the input parameters
        try:
            self.tag_fam_hamming_dist = int(self.tag_family.split('h')[1])
            self.tag_fam_data_bits = int((self.tag_family.split('h')[0]).split('AcTag')[1])
        except ValueError:
            raise ValueError("AcTag family must be in the format " + 
                             "'AcTag<#data_bits>h<#hamming_dist>', eg. AcTag24h8")
        self.tag_diag = np.sqrt(2 * tag_size ** 2)
        self.tag_area = tag_size ** 2
        self.tags_in_family = self.get_tags_in_family(self.path_to_tag_family)
    
    # TODO: Clean up this function
    def get_tags_in_family(self, file_path: str) -> list[list]:
        '''
        This function parses the data from the AcTag family file and returns a list containing the data bit layouts
        for each tag in the family.

        :param file_path: The path to the AcTag family file.
        :type file_path: str
        :return: A list containing the data bit layouts for each tag in the family, including rotations and reversals.
        :rtype: list[list]
        '''

        error_message = "This AcTag family doesn't meet specification for use with the actag repository. " + \
                    "Please refer to the README in the actag_families folder and make sure that you " + \
                    "generate a family with the AcTag family generator."

        # Check that the number of data bits is divisible by 4
        if(self.tag_fam_data_bits % 4 != 0): raise ValueError(error_message)

        # ===== AcTag family is valid, so start parsing the data =====

        # Open the java file and read the contents
        try:
            java_file = open(file_path, "r")
        except FileNotFoundError:
            raise FileNotFoundError("No such file or directory: '" + file_path + "'. Please refer to the README in the actag_families folder for information on how to generate this AcTag family.")
        java_file_str = java_file.read()

        # Close the java file
        java_file.close()

        # Parse the number of tags within the family
        num_tags_in_family = java_file_str[(java_file_str.find("= new long[") + 11):]
        num_tags_in_family = int(num_tags_in_family[:num_tags_in_family.find("]")].strip())

        # For each tag, parse the data bits and put it into a list
        all_tags_lut = []
        curr_file_str = java_file_str
        for i in range(0, num_tags_in_family):
            curr_tag_data_bits = []

            # Find the index of 0x in the file
            data_bit_start_index = curr_file_str.find("0x")

            # Find the index of UL in the file
            data_bit_end_index = curr_file_str.find("L")

            # Extract the data bits from the file as a string
            data_bits_str = curr_file_str[(data_bit_start_index + 2):data_bit_end_index]

            # For each data bit, save it into an array
            current_binary = ""
            for j in range(0, self.tag_fam_data_bits):
                # If current_binary is empty, turn the next hexadecimal value
                # into binary and add it to current_binary
                if(current_binary == ""):
                    current_binary += bin(int(data_bits_str[(len(data_bits_str)-1):], 16))[2:].zfill(4)
                    data_bits_str = data_bits_str[0:(len(data_bits_str)-1)]

                # Save the next binary value into curr_tag_data_bits
                curr_tag_data_bits.insert(0, int(current_binary[(len(current_binary) - 1):]))

                # Remove the last binary value from current_binary
                current_binary = current_binary[0:(len(current_binary) - 1)]
            
            # Add all of the data bits for this tag to the all_tags_lut array
            all_tags_lut.append(curr_tag_data_bits)

            # Substr the curr_file_str for the next tag
            curr_file_str = curr_file_str[(data_bit_end_index + 2):]

        # Get the number of data bits per side
        data_bits_strPerSide = self.tag_fam_data_bits / 4

        # Create rotations for each tag and add it to the addTags array
        # Do this three times, for 90, 180, and 270 degrees
        for i in range(1, 4):
            # For each tag in the all_tags_lut array
            for j in range(0, num_tags_in_family):
                rotating_tag = copy.deepcopy(all_tags_lut[j:j+1][0])
                # Rotate entries in the tag data_bits_strPerSide*i times
                for k in range(0, int(data_bits_strPerSide*i)):
                    rotating_tag.append(rotating_tag.pop(0))
                all_tags_lut.append(rotating_tag)

        # Create another copy of each tag, but this one goes counter-clockwise instead of clockwise
        for i in range(0, len(all_tags_lut)):
            ccTag = copy.deepcopy(all_tags_lut[i])
            for j in range(1, int(len(ccTag) / 2)):
                temp = ccTag[j]
                ccTag[j] = ccTag[len(ccTag) - j]
                ccTag[len(ccTag) - j] = temp
            all_tags_lut.append(ccTag)

        # Return the all_tags_lut array
        return all_tags_lut
    
def main():
     ## Example class setup
    tag_family = "AcTag24h10"
    tag_size = 3 / 7 # For 1 meter (outer dimensions) tag w/ 24 data bits.
    tagp = AcTagParams(tag_family, tag_size)
   
if __name__ == "__main__":
    main()