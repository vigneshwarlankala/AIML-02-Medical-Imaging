from pathlib import Path
import struct
from datetime import datetime
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    generate_uid
)

output = Path("outputs/dicom_test")
output.mkdir(parents=True, exist_ok=True)
filename = output / "synthetic_ct_test.dcm"

study_uid = generate_uid()
series_uid = generate_uid()
sop_uid = generate_uid()

meta = FileMetaDataset()
meta.MediaStorageSOPClassUID = CTImageStorage
meta.MediaStorageSOPInstanceUID = sop_uid
meta.TransferSyntaxUID = ExplicitVRLittleEndian
meta.ImplementationClassUID = generate_uid()

ds = FileDataset(str(filename), {}, file_meta=meta, preamble=b"\0" * 128)

now = datetime.now()
ds.SOPClassUID = CTImageStorage
ds.SOPInstanceUID = sop_uid
ds.StudyInstanceUID = study_uid
ds.SeriesInstanceUID = series_uid
ds.Modality = "CT"
ds.PatientName = "Synthetic^Test"
ds.PatientID = "SYNTHETIC001"
ds.StudyDescription = "Synthetic DICOM Storage Test"
ds.SeriesDescription = "Test CT Series"
ds.StudyDate = now.strftime("%Y%m%d")
ds.StudyTime = now.strftime("%H%M%S")
ds.Rows = 16
ds.Columns = 16
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"
ds.BitsAllocated = 16
ds.BitsStored = 16
ds.HighBit = 15
ds.PixelRepresentation = 0
ds.InstanceNumber = 1
ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
ds.PixelData = struct.pack("<256H", *[i * 16 for i in range(256)])

ds.is_little_endian = True
ds.is_implicit_VR = False
ds.save_as(filename, enforce_file_format=True)

print("Synthetic DICOM created:", filename.resolve())
print("File size:", filename.stat().st_size, "bytes")
print("Patient ID:", ds.PatientID)
print("Modality:", ds.Modality)
print("Dimensions:", ds.Rows, "x", ds.Columns)
