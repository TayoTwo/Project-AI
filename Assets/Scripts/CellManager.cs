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

        //This variable is used later when spawning the stage to have the center of the grid be at Vector.zero
        offset = new Vector3(gridDim.x,0,gridDim.y) * 0.5f * unitLength;
        grid = new Cell[gridDim.x,gridDim.y];

        //Loop through the grid and create a Cell class at every position
        for(int x = 0;x < gridDim.x;x++){

            for(int y = 0;y < gridDim.y;y++){

                //When translating the grid position to world space set the center of the grid to (0,0) by shifting it by an offset
                Vector3 pos = new Vector3(x * unitLength + (unitLength/2) ,0,y * unitLength + (unitLength/2) ) - offset;
                //Check if there is an obstacle at this cells position, if so then set the cell as not walkable
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

        //We are checking the 8 neighbouring cells so we look in a 3x3 area around the original cell
        for(int x = -1;x < 2;x++){

            for(int y = -1;y < 2;y++){

                //If referencing the original cells position
                if(x == 0 && y == 0){

                    continue;

                }

                //Neighbours grid position
                int neighX = n.gridPos.x + x;
                int neighY = n.gridPos.y + y;

                //If the neighbours position is within the grid
                if(neighX >= 0 && neighY >= 0 && neighX < gridDim.x && neighY < gridDim.y){

                    neighbours.Add(grid[neighX,neighY]);

                }

            }

        }

        return neighbours;

    }

    void OnDrawGizmos(){

        //Draw a cube showing the pathfinding space
        Gizmos.DrawWireCube(transform.position,new Vector3(gridDim.x,1,gridDim.y) * unitLength);

        if(grid != null){

            //Look through every cell and visually show if it is walkable or not
            foreach(Cell cell in grid){

                if(cell.walkable){

                    Gizmos.color = Color.green;

                } else {

                    Gizmos.color = Color.red;

                }

                Gizmos.DrawCube(cell.pos,Vector3.one * unitLength * 0.9f);

            }

        }

    }
}
