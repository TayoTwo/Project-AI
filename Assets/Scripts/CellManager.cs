using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Cell {

    public bool walkable;
    public Vector3 pos;
    public Vector2Int gridPos;
    public int g;
    public int h;
    public Cell parent;

    public Cell(bool w,Vector3 p,Vector2Int gPos){

        walkable = w;
        pos = p;
        gridPos = gPos;

    }

    public int F(){

        return g + h;

    }

}

public class CellManager : MonoBehaviour{

    public LayerMask layerMask;
    public Vector2Int gridDim;
    public float unitLength;
    public Cell[,] grid;
    Vector3 offset;

    //Setup the grid
    public void Initialize(){

        offset = new Vector3(gridDim.x,0,gridDim.y) * 0.5f * unitLength;
        grid = new Cell[gridDim.x,gridDim.y];
        //This variable is used later when spawning the stage to have the center of the grid be at Vector.zero (not used on the walkers)

        //Loop through the grid and create a Cell class at every position
        for(int x = 0;x < gridDim.x;x++){

            for(int y = 0;y < gridDim.y;y++){

                Vector3 pos = new Vector3(x * unitLength + (unitLength/2) ,0,y * unitLength + (unitLength/2) ) - offset;
                bool isWalkable = !(Physics.CheckSphere(pos,unitLength/2f,layerMask));

                grid[x,y] = new Cell(isWalkable,pos,new Vector2Int(x,y));

            }

        }

    }

    //Translate a world position to the nearest cell in the grid
    public Cell WorldPosToCell(Vector3 pos){

        pos += offset;
        pos /= unitLength;
        pos = new Vector3(Mathf.Clamp(pos.x,0,gridDim.x),0,Mathf.Clamp(pos.z,0,gridDim.y));
        Vector3Int posN = Vector3Int.RoundToInt(pos); 

        return grid[posN.x,posN.z];

    }

    public List<Cell> GetNeighbours(Cell n){

        List<Cell> neighbours = new List<Cell>();

        for(int x = -1;x < 2;x++){

            for(int y = -1;y < 2;y++){

                if(x == 0 && y == 0){

                    continue;

                }

                int neighX = n.gridPos.x + x;
                int neighY = n.gridPos.y + y;

                if(neighX >= 0 && neighY >= 0 && neighX < gridDim.x && neighY < gridDim.y){

                    //Debug.Log("X: " + neighX + " Y: " + neighY);
                    neighbours.Add(grid[neighX,neighY]);

                }

            }

        }

        return neighbours;

    }

    void OnDrawGizmos(){

        Gizmos.DrawWireCube(transform.position,new Vector3(gridDim.x,1,gridDim.y) * unitLength);

        if(grid != null){

            foreach(Cell n in grid){

                if(n.walkable){

                    Gizmos.color = Color.green;

                } else {

                    Gizmos.color = Color.red;

                }

                Gizmos.DrawCube(n.pos,Vector3.one * unitLength * 0.9f);

            }

        }

    }
}
