Merge "brain-h3.0.stl";

// Define surface loop and volume
Surface Loop(1) = {1}; // or more surfaces if needed
Volume(1) = {1};

// Mesh settings
Mesh.CharacteristicLengthMin = 1;
Mesh.CharacteristicLengthMax = 2;

// 3D mesh
Mesh 3;

// Save to MSH version 2
Save "brain-h3.0.msh";
