import unittest
import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from src.application import Base, FaceData

class TestDBSchema(unittest.TestCase):

    def setUp(self):
        # Create an in-memory SQLite database for testing
        self.engine = create_engine('sqlite:///:memory:')

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Create a session factory
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def tearDown(self):
        # Drop all tables
        Base.metadata.drop_all(self.engine)

    def test_face_data_model(self):
        # Create a test face data entry
        face_id = "test_id_123"
        name = "Test Person"
        registered_at = datetime.now()
        embedding = "serialized_embedding_data"

        face_data = FaceData(
            id=face_id,
            name=name,
            registered_at=registered_at,
            embedding=embedding
        )

        # Add to the session and commit
        self.session.add(face_data)
        self.session.commit()

        # Query and verify
        result = self.session.query(FaceData).filter_by(id=face_id).first()

        self.assertEqual(result.id, face_id)
        self.assertEqual(result.name, name)
        self.assertEqual(result.embedding, embedding)

    def test_face_data_model_constraints(self):
        # Test that id is required (primary key)
        face_data_no_id = FaceData(
            name="No ID Person",
            registered_at=datetime.now()
        )

        # Should raise an exception when committed
        with self.assertRaises(Exception):
            self.session.add(face_data_no_id)
            self.session.commit()

        # Rollback after exception
        self.session.rollback()

        # Test duplicate id (primary key violation)
        face_data1 = FaceData(
            id="duplicate_id",
            name="Person 1",
            registered_at=datetime.now()
        )

        face_data2 = FaceData(
            id="duplicate_id",
            name="Person 2",
            registered_at=datetime.now()
        )

        self.session.add(face_data1)
        self.session.commit()

        # Adding with the same ID should fail
        with self.assertRaises(Exception):
            self.session.add(face_data2)
            self.session.commit()

        # Rollback after exception
        self.session.rollback()

    def test_face_data_queries(self):
        # Add multiple test entries
        test_faces = [
            FaceData(id=f"id_{i}", name=f"Person {i}", registered_at=datetime.now())
            for i in range(5)
        ]

        for face in test_faces:
            self.session.add(face)
        self.session.commit()

        # Test querying by name
        person_2 = self.session.query(FaceData).filter_by(name="Person 2").first()
        self.assertEqual(person_2.id, "id_2")

        # Test getting all faces
        all_faces = self.session.query(FaceData).all()
        self.assertEqual(len(all_faces), 5)

        # Test deleting a face
        self.session.delete(person_2)
        self.session.commit()

        # Verify deletion
        deleted_check = self.session.query(FaceData).filter_by(name="Person 2").first()
        self.assertIsNone(deleted_check)

        # Verify remaining count
        remaining = self.session.query(FaceData).all()
        self.assertEqual(len(remaining), 4)