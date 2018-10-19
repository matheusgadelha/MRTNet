import java.io.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;

import javax.vecmath.Point3f;
import javax.vecmath.Point3i;
import javax.vecmath.Vector3f;

class cotSpectral {

	/* 'input_verts' and 'input_faces' store the vertex and face data of the input mesh respectively
	 * each element in 'input_verts' is a 3D points defining the vertex
	 * every three integers in 'input_faces' define the indexes of the three vertices that make a triangle
	 * there are in total input_faces.size()/3 triangles
	 */
	private static ArrayList<Point3f> input_verts = new ArrayList<Point3f> ();
	private static ArrayList<Integer> input_faces = new ArrayList<Integer> ();
	
	/* 'curr_verts' and 'curr_faces' store the subdivided mesh that will be displayed on the screen
	 * the elements stored in these arrays are the same with the input mesh
	 * 'curr_normals' stores the normal of each face and is necessary for shading calculation
	 * you don't have to compute the normals yourself: once you have curr_verts and curr_faces
	 * ready, you just need to call estimateFaceNormal to generate normal data
	 */
	private static ArrayList<Point3f> curr_verts = new ArrayList<Point3f> ();
	private static ArrayList<Integer> curr_faces = new ArrayList<Integer> ();
	private static ArrayList<Vector3f> curr_normals = new ArrayList<Vector3f> ();
	
	private static String dmetric = "geodesic";
	private static String sortmethod = "kd";
	
	private static float xmin, ymin, zmin;
	private static float xmax, ymax, zmax;
	private static String output_filename;

	private static ArrayList<Point3f> samples = null;
	private static ArrayList<ASPPoint> aspSamples = null;
	private static float cum_area[];
	private static Vector3f normals[];
	
	//minimum radius points must be apart from one another.
	private static float radius = 0.f, radiusSq = radius*radius;
	private static float rho = 0.75f;
	private static int NSamples = 256;
	
	/* load a simple .obj mesh from disk file
	 * note that each face *must* be a triangle and cannot be a quad or other types of polygon
	 */ 
	private static void loadMesh(String filename) {
		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader(filename));
		} catch (IOException e) {
			System.out.println("Error reading from file " + filename);
			System.exit(0);
		}

		float x, y, z;
		int v1, v2, v3;
		//int dummy;
		String line;
		String[] tokens;
		String[] subtokens;
		try {
		while ((line = in.readLine()) != null) {
			if (line.length() == 0)
				continue;
			switch(line.charAt(0)) {
			case 'v':
				if (line.charAt(1) == 't') {
					continue;	// do not process texture coords
				}
				if (line.charAt(1) == 'n') {
					continue;	// do not process normal					
				}
				tokens = line.split("[ ]+");
				x = Float.valueOf(tokens[1]);
				y = Float.valueOf(tokens[2]);
				z = Float.valueOf(tokens[3]);
				input_verts.add(new Point3f(x, y, z));
				break;
			case 'f':
				tokens = line.split("[ ]+");
				/* when defining faces, .obj assumes the vertex index starts from 1
				 * so we should subtract 1 from each index 
				 */
				subtokens = tokens[1].split("/+");
				v1 = Integer.valueOf(subtokens[0])-1;
				
				subtokens = tokens[2].split("/+");
				v2 = Integer.valueOf(subtokens[0])-1;
				
				subtokens = tokens[3].split("/+");
				v3 = Integer.valueOf(subtokens[0])-1;
				
				input_faces.add(v1);
				input_faces.add(v2);
				input_faces.add(v3);				
				break;
			default:
				continue;
			}
		}
		in.close();	
		} catch(IOException e) {
			// error reading file
		}

		//System.out.println("Read " + input_verts.size() + " vertices and " + input_faces.size() + " faces.");
		System.out.print(filename+" ");
		
		int tris = input_faces.size()/3;
		cum_area = new float[tris];
		normals = new Vector3f[tris];
		float cur_area = 0;
		for(int i=0; i<tris; i++){
			Point3f p1 = input_verts.get(input_faces.get(3*i));
			Point3f p2 = input_verts.get(input_faces.get(3*i+1));
			Point3f p3 = input_verts.get(input_faces.get(3*i+2));
			Vector3f a = new Vector3f(), b = new Vector3f();
			a.sub(p2, p1); b.sub(p3, p1);
			Vector3f res = new Vector3f(); res.cross(a, b);
			float len = res.length();
			cur_area += 0.5f*len;
			res.scale(1.f/len);
			normals[i] = res;
			cum_area[i] = cur_area;
		}

	}
	
	/* find the bounding box of all vertices */
	private static void computeBoundingBox() {
		xmax = xmin = input_verts.get(0).x;
		ymax = ymin = input_verts.get(0).y;
		zmax = zmin = input_verts.get(0).z;
		
		for (int i = 1; i < input_verts.size(); i ++) {
			xmax = Math.max(xmax, input_verts.get(i).x);
			xmin = Math.min(xmin, input_verts.get(i).x);
			ymax = Math.max(ymax, input_verts.get(i).y);
			ymin = Math.min(ymin, input_verts.get(i).y);
			zmax = Math.max(zmax, input_verts.get(i).z);
			zmin = Math.min(zmin, input_verts.get(i).z);			
		}
		
		// radius = 0.02f*Math.max(xmax-xmin, Math.max(ymax-ymin, zmax-zmin));
		// Select radius
		float total_area = cum_area[cum_area.length-1];
		radius = (float)(rho*Math.pow(3.464101615f*NSamples/total_area, -0.5f)*2);
		radiusSq = radius*radius;
		System.out.println("radius = " + radius);
	}

	public static void printUsage() {
		System.out.println("Usage: java -cp .:./vecmath.jar cotSpectral in.obj out.obj nsamples [sorting] [dmetric]");
		System.exit(1);
	}

	public static void main(String[] args) {

		String inputFilename = null;
		if (args.length < 3) {
			printUsage();
		}
		inputFilename = args[0];
		output_filename = args[1];
		NSamples = Integer.parseInt(args[2]); 

		if(args.length>=4)	sortmethod = args[3];
		if(args.length>=5)	dmetric = args[4];

		loadMesh(inputFilename);
		computeBoundingBox();

		curr_verts = input_verts;
		curr_faces = input_faces;
		estimateFaceNormal();

		for (int i=0; i<1; ++i)
		{
			while(generateSamples()==false) {
				rho-=0.05;
				float total_area = cum_area[cum_area.length-1];
				radius = (float)(rho*Math.pow(3.464101615f*NSamples/total_area, -0.5f)*2);
				radiusSq = radius*radius;
			}
			sortASPSamples();
			saveSamples(true, i);
		}
	}

	private static String addId(String filepath, int id){
		return filepath.replaceAll(".obj", "_"+String.valueOf(id)+".obj");
	}

	private static void saveSamples(boolean write_normal, int id){
		String samples_filename = addId(output_filename, id);
		File f = new File(samples_filename);
		f.getParentFile().mkdirs();
		try{
			BufferedWriter w = new BufferedWriter(new FileWriter(samples_filename));
			for(ASPPoint s : aspSamples) {
				w.write("v "+s.point.x+" "+s.point.y+" "+s.point.z+"\n");
				if(write_normal) w.write("vn "+s.normal.x+" "+s.normal.y+" "+s.normal.z+"\n");
			}
			w.flush();
			w.close();
		} catch(Exception e){e.printStackTrace();}	
		
		System.out.println(samples_filename+" saved");
	}
	
	private static class AuxComparatorX implements Comparator<ASPPoint> {
		public int compare(ASPPoint o1, ASPPoint o2) {
			if (o1.point.x > o2.point.x) return 1; if (o1.point.x < o2.point.x) return -1; return 0;
		}
	}
	private static class AuxComparatorY implements Comparator<ASPPoint> {
		public int compare(ASPPoint o1, ASPPoint o2) {
			if (o1.point.y > o2.point.y) return 1; if (o1.point.y < o2.point.y) return -1; return 0;
		}
	}
	private static class AuxComparatorZ implements Comparator<ASPPoint> {
		public int compare(ASPPoint o1, ASPPoint o2) {
			if (o1.point.z > o2.point.z) return 1; if (o1.point.z < o2.point.z) return -1; return 0;
		}
	}
	private static class AuxComparatorXYZ implements Comparator<ASPPoint> {
		public int compare(ASPPoint o1, ASPPoint o2) {
			float xyz1 = o1.point.x+o1.point.y+o1.point.z;
			float xyz2 = o2.point.x+o2.point.y+o2.point.z;
			if (xyz1 > xyz2) return 1; if (xyz1 < xyz2) return -1; return 0;
		}
	}
	private static void sortASPSamples() {
		ASPPoint[] pts = new ASPPoint[aspSamples.size()];
		for(int i=0;i<pts.length;i++) pts[i]=aspSamples.get(i);
		
		if(sortmethod.equals("xyz")) {
			Arrays.sort(pts, 0, pts.length, new AuxComparatorXYZ());	//sort with x+y+z
		} else {
			recursiveSortAspSamples(0, pts.length, pts, 0);				//sort with kd
		}
		aspSamples = new ArrayList<ASPPoint>(Arrays.asList(pts));
	}
	private static void recursiveSortAspSamples(int start, int end, ASPPoint[] pts, int axis) {
		if(end-start<=1) return;
		if(axis==0)
			Arrays.sort(pts, start, end, new AuxComparatorX());
		else if(axis==1)
			Arrays.sort(pts, start, end, new AuxComparatorY());
		else
			Arrays.sort(pts, start, end, new AuxComparatorZ());
		int mid=(start+end)/2;
		recursiveSortAspSamples(start, mid, pts, (axis+1)%3);
		recursiveSortAspSamples(mid, end, pts, (axis+1)%3);
	}
	
	private static class Grid {
		private static final int GRID_SIZE = 32;
		private final HashMap<Point3i, ArrayList<ASPPoint>> grid = new HashMap<Point3i, ArrayList<ASPPoint>>();
		public Grid(){}
		public void insert(ASPPoint p){
			Point3i g = new Point3i();
			g.x = (int)(((p.point.x-xmin)/(xmax-xmin))*GRID_SIZE);
			g.y = (int)(((p.point.y-ymin)/(ymax-ymin))*GRID_SIZE);
			g.z = (int)(((p.point.z-zmin)/(zmax-zmin))*GRID_SIZE);
			ArrayList<ASPPoint> pts = grid.get(g);
			if(pts == null){pts = new ArrayList<ASPPoint>(); grid.put(g, pts);}
			pts.add(p);
		}
		public boolean legalPoint(ASPPoint p){
			Point3i min = new Point3i();
			Point3i max = new Point3i();			
		
			min.x = (int)(((p.point.x-radius-xmin)/(xmax-xmin))*GRID_SIZE);
			min.y = (int)(((p.point.y-radius-ymin)/(ymax-ymin))*GRID_SIZE);
			min.z = (int)(((p.point.z-radius-zmin)/(zmax-zmin))*GRID_SIZE);

			max.x = (int)(((p.point.x+radius-xmin)/(xmax-xmin))*GRID_SIZE);
			max.y = (int)(((p.point.y+radius-ymin)/(ymax-ymin))*GRID_SIZE);
			max.z = (int)(((p.point.z+radius-zmin)/(zmax-zmin))*GRID_SIZE);

			if (xmax == xmin) {
				min.x = 0;
				max.x = 0;
			}
			if (ymax == ymin) {
				min.y = 0;
				max.y = 0;
			}
			if (zmax == zmin) {
				min.z = 0;
				max.z = 0;
			}	

			Point3i g = new Point3i();
			for(g.x = min.x; g.x <= max.x; g.x++)
				for(g.y = min.y; g.y <= max.y; g.y++)
					for(g.z = min.z; g.z <= max.z; g.z++){
						ArrayList<ASPPoint> pts = grid.get(g);
						if(pts != null)
							for(ASPPoint s : pts){
								float distSq = s.point.distanceSquared(p.point);
								if(distSq > radiusSq) continue;
								float scaling_factor = 0.f;
								if(dmetric.equals("random"))  scaling_factor = 10000.f;
								else scaling_factor = getScalingFactor(s, p);
								if(distSq*scaling_factor < radiusSq) return false;
							}
					}
			return true;
		}
		private void clearGrid(){
			grid.clear();
		}
	}
	
	private static float getScalingFactor(ASPPoint s, ASPPoint p){
		float scaling_factor = 0;
		Vector3f D = new Vector3f(s.point.x - p.point.x, s.point.y - p.point.y, s.point.z - p.point.z);
		
		D.normalize();		
		float a = D.dot(s.normal), b = D.dot(p.normal);
		
		
		if (a == b)	scaling_factor = (float)(1.0 / Math.sqrt(1.0 - a * a));
		else scaling_factor = (float)((Math.asin(a) - Math.asin(b)) / (a - b));
		return (float)scaling_factor * scaling_factor;
		
		/*
		if(a*b >= 0) return 1;
		a = Math.abs(a); b = Math.abs(b);
		float total = a+b;
		scaling_factor =  a/total*1.f/(float)Math.sqrt(1 - a*a)+b/total*1.f/(float)Math.sqrt(1 - b*b);
		System.out.println(scaling_factor);		
		return scaling_factor * scaling_factor;*/
		
		
/*
		 for (int i = 0; i < 21; i++) {
			 float t = (i / 20.f);
			 float tPow = (float)Math.pow(t, 20);
			 float tm1Pow = (float)Math.pow(1-t, 20);
			 //sample the function at t and add f(t)*0.05;
			 Vector3f N = new Vector3f(
			   tPow*s.normal.x + tm1Pow*p.normal.x,
			   tPow*s.normal.y + tm1Pow*p.normal.y,
			   tPow*s.normal.z + tm1Pow*p.normal.z
			   );
			 N.normalize();
			 float ND = N.dot(D);
			 float f = 1.f / (float)Math.sqrt(1 - ND*ND);
			 scaling_factor += (f*0.0476190476f);


			 }
		 System.out.println(scaling_factor);
		 return scaling_factor * scaling_factor;
		 */
		
	}
	
	private static final Grid grid = new Grid();
	
	private static class ASPPoint {
		public final Point3f point;
		public final Vector3f normal;
		public final int v1, v2, v3;
		public final float l1, l2;
		public ASPPoint(Vector3f n, int v1, int v2, int v3, float l1, float l2){
			normal = n; this.v1 = v1; this.v2 = v2; this.v3 = v3; this.l1 = l1; this.l2 = l2;
			Point3f p1 = curr_verts.get(v1);
			Point3f p2 = curr_verts.get(v2);
			Point3f p3 = curr_verts.get(v3);
			float l3 = 1.f-l1-l2;
			point = new Point3f(l1*p1.x+l2*p2.x+l3*p3.x, l1*p1.y+l2*p2.y+l3*p3.y, l1*p1.z+l2*p2.z+l3*p3.z);
		}
		/*public void write(Writer w) throws Exception {w.write(v1+" "+v2+" "+v3+" "+l1+" "+l2+"\n");}
		public void writeobj(Writer w) throws Exception {
			w.write("v "+point.x+" "+point.y+" "+point.z+"\n");
		}*/
		
	}
	
	private static int getIdx(int i){return (i<0)?-i-1:i;}
	
	private static boolean generateSamples(){
		float cur_area = cum_area[cum_area.length-1];
		samples = new ArrayList<Point3f> ();
		aspSamples = new ArrayList<ASPPoint>();
		grid.clearGrid(); System.gc();
		boolean foundPoint = true;
		int minTries = 1000;
		while(foundPoint){
			foundPoint = false;
			if(samples.size() >= NSamples) break;
			
			int actualNumTries = Math.max(minTries, samples.size() * 15);
			for(int i=0; i<actualNumTries; i++){
				Random r = new Random();
				float f = Math.min(r.nextFloat()*cur_area, cur_area);
				int tri = getIdx(Arrays.binarySearch(cum_area, f));
				float u = (float)Math.sqrt(r.nextFloat());
				float l1 = 1.f-u, l2 = r.nextFloat()*u;
				ASPPoint next = new ASPPoint(normals[tri], curr_faces.get(3*tri), curr_faces.get(3*tri+1), curr_faces.get(3*tri+2), l1, l2);
				if(grid.legalPoint(next)){grid.insert(next); aspSamples.add(next); samples.add(next.point); foundPoint = true; break;}
			}
		}
		System.out.println("number of samples: "+samples.size()+" (rho="+rho+")");
		if(samples.size()==NSamples) return true;
		else return false;
	}
	
	/* estimate face normals */
	private static void estimateFaceNormal() {
		int i;
		curr_normals.clear();
		for (i = 0; i < curr_faces.size(); i ++) {
			curr_normals.add(new Vector3f());
		}
		
		Vector3f e1 = new Vector3f();
		Vector3f e2 = new Vector3f();
		for (i = 0; i < curr_faces.size()/3; i ++) {
			// get face
			int v1 = curr_faces.get(3*i+0);
			int v2 = curr_faces.get(3*i+1);
			int v3 = curr_faces.get(3*i+2);
			
			// compute normal
			e1.sub(curr_verts.get(v2), curr_verts.get(v1));
			e2.sub(curr_verts.get(v3), curr_verts.get(v1));
			curr_normals.get(i*3+0).cross(e1, e2);
			curr_normals.get(i*3+0).normalize();
			curr_normals.get(i*3+1).cross(e1, e2);
			curr_normals.get(i*3+1).normalize();
			curr_normals.get(i*3+2).cross(e1, e2);
			curr_normals.get(i*3+2).normalize();
		}
	}
}
